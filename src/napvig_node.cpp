#include "napvig_node.h"
#include "multi_array_manager.h"

#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Pose2D.h>
#include <nav_msgs/Path.h>
#include <napvig/SearchHistory.h>

using namespace ros;
using namespace std;
using namespace XmlRpc;
using namespace torch;
using namespace at;
using namespace torch::indexing;

Tensor quaternionMsgToTorch(const geometry_msgs::Quaternion &quaternionMsg) {
	return torch::tensor ({quaternionMsg.x,
						   quaternionMsg.y,
						   quaternionMsg.z,
						   quaternionMsg.w}, torch::kDouble);
}
Tensor pointMsgToTorch (const geometry_msgs::Point &vectorMsg) {
	return torch::tensor ({vectorMsg.x,
						  vectorMsg.y,
						  vectorMsg.z}, torch::kDouble);
}

NapvigNode::NapvigNode ():
	SparcsNode(NODE_NAME)
{
	initParams ();
	initROS ();

	initTestGrid ();
}

void NapvigNode::initTestGrid()
{
	Tensor xyRange = torch::arange (nodeParams.mapTestRangeMin, nodeParams.mapTestRangeMax, nodeParams.mapTestRangeStep, torch::dtype (kFloat64));
	Tensor xx, yy;
	vector<Tensor> xy;

	xy = meshgrid ({xyRange, xyRange});

	xx = xy[0].reshape (-1);
	yy = xy[1].reshape (-1);

	testGrid.points = stack ({xx, yy}, 1);
	testGrid.xySize = xyRange.size (0);
}

void NapvigNode::initParams ()
{
	NapvigMap::Params mapParams;
	Napvig::Params napvigParams;

	mapParams.measureRadius = paramDouble (params["map"], "measure_radius");
	mapParams.smoothRadius = paramDouble (params["map"], "smooth_radius");
	mapParams.precision = paramInt (params["map"],"precision");
	mapParams.dim = paramInt(params["map"],"dimensions");

	napvigParams.stepAheadSize = paramDouble (params["napvig"], "step_ahead_size");
	napvigParams.gradientStepSize = paramDouble (params["napvig"], "gradient_step_size");
	napvigParams.terminationDistance = paramDouble (params["napvig"], "termination_distance");
	napvigParams.terminationCount = paramInt (params["napvig"], "termination_count");
	napvigParams.lookaheadHorizon = paramInt  (params["napvig"], "lookahead_horizon");
	napvigParams.algorithm = paramEnum<Napvig::AlgorithmType> (params["napvig"], "algorithm", {"single_step", "randomized_recovery","optimized_trajectory"});
	napvigParams.minDistance = paramDouble (params["napvig"],"min_distance");
	napvigParams.scatterVariance = paramDouble (params["napvig"], "scatter_variance");
	napvigParams.keepLastSearch = paramBool (params["napvig"],"keep_last_search");
	if (napvigParams.algorithm == Napvig::OPTIMIZED_TRAJECTORY) {
		napvigParams.trajectoryOptimizerParams.rangeAngleMin = paramDouble (params["napvig"]["trajectory_optimizer"],"scan_range_min");
		napvigParams.trajectoryOptimizerParams.rangeAngleMax = paramDouble (params["napvig"]["trajectory_optimizer"],"scan_range_max");
		napvigParams.trajectoryOptimizerParams.rangeAngleStep = paramDouble (params["napvig"]["trajectory_optimizer"],"scan_range_step");
	}

	nodeParams.mapTestRangeMin = paramDouble (params["map_test"], "range_min");
	nodeParams.mapTestRangeMax = paramDouble (params["map_test"], "range_max");
	nodeParams.mapTestRangeStep = paramDouble (params["map_test"], "range_step");
	nodeParams.drawWhat = paramEnum<TestDraw> (params["map_test"],"draw", {"none","value","grad","minimal"});

	napvig = new Napvig (mapParams, napvigParams, &debug);
}

void NapvigNode::initROS ()
{
	addSub ("measures_sub", paramString (params["topics"]["subs"],"scan"), 1, &NapvigNode::measuresCallback);
	addSub ("odom_sub", paramString (params["topics"]["subs"], "odom"), 5, &NapvigNode::odomCallback);
	addSub ("target_sub", paramString (params["topics"]["subs"], "target"), 1, &NapvigNode::targetCallback);
	addSub ("corridor_sub", "/corridor", 1, &NapvigNode::corridorCallback);

	addPub<std_msgs::Float64MultiArray> ("measures_pub", paramString(params["topics"]["pubs"],"measures"),1);
	addPub<std_msgs::Float64MultiArray> ("map_values_pub", paramString(params["topics"]["pubs"],"map_values"),1);
	addPub<geometry_msgs::Pose2D> ("setpoint_pub", paramString(params["topics"]["pubs"],"setpoint"), 1);
	addPub<std_msgs::Float64MultiArray> ("grad_log_pub","/grad_log", 1);
	addPub<napvig::SearchHistory> ("search_history_pub", paramString (params["topics"]["pubs"],"search_history"), 1);
	addPub<geometry_msgs::Pose2D> ("world_setpoint_pub","/setpoint_world", 1);
	addPub<std_msgs::Float64> ("corridor_distance_pub","/corridor_distance", 1);
}

Tensor polar2rectangularMeasure (double radius, double angle) {
	return torch::tensor ({radius * cos(angle), radius*sin(angle)});
}

Tensor NapvigNode::convertScanMsg (const sensor_msgs::LaserScan &scanMsg)
{
	const int measCount = scanMsg.ranges.size ();
	Tensor measures = torch::empty ({measCount, N_DIM});

	for (int i = 0; i < measCount; i++) {
		double currAngle = scanMsg.angle_min + i * scanMsg.angle_increment;
		double currRadius = scanMsg.ranges[i];

		measures.index_put_ ({i, None}, polar2rectangularMeasure (currRadius, currAngle));
	}

	return measures;
}

void NapvigNode::publishMeasures (const torch::Tensor &measures)
{
	std_msgs::Float64MultiArray measuresMsg;
	double *measData = (double *) measures.toType (ScalarType::Double).data_ptr();

	measuresMsg.layout.dim.resize (2);
	measuresMsg.layout.dim[0].size = measures.size (0);
	measuresMsg.layout.dim[1].size = measures.size (1);

	measuresMsg.data = vector<double> (measData, measData + measures.numel ());

	publish ("measures_pub", measuresMsg);
}

#define RANGE_DIM 3

void NapvigNode::publishValues ()
{
	if (!napvig->isMapReady () || nodeParams.drawWhat == TEST_DRAW_NONE || nodeParams.drawWhat == TEST_DRAW_MINIMAL)
		return;

	bool grad = (nodeParams.drawWhat == TEST_DRAW_GRAD);

	std_msgs::Float64MultiArray mapValuesMsg;
	const int dimensions = grad? 3: 2;
	mapValuesMsg.layout.dim.resize (dimensions);
	mapValuesMsg.layout.dim[0].size = testGrid.xySize;
	mapValuesMsg.layout.dim[1].size = testGrid.xySize;
	if (grad)
		mapValuesMsg.layout.dim[2].size = N_DIM;

	mapValuesMsg.layout.data_offset = RANGE_DIM;
	mapValuesMsg.data.resize (RANGE_DIM + (grad?2:1)*testGrid.points.size (0));

	mapValuesMsg.data[0] = nodeParams.mapTestRangeMin;
	mapValuesMsg.data[1] = nodeParams.mapTestRangeMax;
	mapValuesMsg.data[2] = nodeParams.mapTestRangeStep;

	for (int i = 0; i < testGrid.points.size (0); i++) {
		Tensor currPoint = testGrid.points.index ({i, None});

		if (grad) {
			Tensor grad = napvig->mapGrad (currPoint);

			mapValuesMsg.data[RANGE_DIM + 2*i] = grad[0].item ().toDouble ();
			mapValuesMsg.data[RANGE_DIM + 2*i + 1] = grad[1].item ().toDouble ();
		} else
			mapValuesMsg.data[RANGE_DIM + i] = napvig->mapValue (currPoint);
	}

	publish ("map_values_pub", mapValuesMsg);
}

void NapvigNode::corridorCallback (const std_msgs::Float32MultiArray &corridorMsg)
{
	MultiArray32Manager array(corridorMsg);
	torch::Tensor corridorTensor;

	corridorTensor = torch::zeros ({array.size (0), array.size (1)}, kDouble);

	for (int i = 0; i < array.size (0); i++)
		for (int j = 0; j < array.size (1); j++)
			corridorTensor[i][j] = array.get ({i,j});
	napvig->setCorridor (corridorTensor);
}

void NapvigNode::measuresCallback (const sensor_msgs::LaserScan &scanMsg)
{
	torch::Tensor newMeasures = convertScanMsg (scanMsg);

	napvig->setMeasures (newMeasures);
	publishMeasures (newMeasures);
}

void NapvigNode::odomCallback (const nav_msgs::Odometry &odomMsg) {
	napvig->updateFrame (Frame{Rotation (quaternionMsgToTorch (odomMsg.pose.pose.orientation)),
							   pointMsgToTorch (odomMsg.pose.pose.position).slice (0,0,2)});
	worldPos = torch::tensor ({odomMsg.pose.pose.position.x,
							  odomMsg.pose.pose.position.y},kDouble);
	worldOrient = quaternionMsgToTorch (odomMsg.pose.pose.orientation);

	syncActions ();
}

void NapvigNode::targetCallback (const geometry_msgs::Pose &targetMsg) {
	napvig->updateTarget (Frame{Rotation (quaternionMsgToTorch (targetMsg.orientation)),
								pointMsgToTorch (targetMsg.position).slice (0,0,2)});
}

void NapvigNode::publishControl ()
{
	if (!napvig->isReady ())
		return;

	geometry_msgs::Pose2D poseMsg, worldPoseMsg;
	torch::Tensor nextStep, nextBearing, worldStep;

	if (!napvig->step ()) {
		ROS_ERROR ("NO STEP");
		return;
	} else {
		ROS_INFO ("YES STEP");
	}

	nextStep = napvig->getSetpointPosition ();
	nextBearing = napvig->getSetpointDirection ();

	poseMsg.x = nextStep[0].item ().toDouble ();
	poseMsg.y = nextStep[1].item ().toDouble ();
	poseMsg.theta = atan2 (poseMsg.y, poseMsg.x);
	worldStep = nextStep + worldPos;


	/************************************
	 * TACCONE INCREDIBILE CORREGGERE ASAP
	 * SE NON LO FAI GUARDA TI AMMAZZO MALE
	 * *********************************/
	Eigen::Vector3d pos3d, worldFrameTot;
	Eigen::Quaterniond quat(worldOrient[3].item().toDouble (),
			worldOrient[0].item().toDouble (),
			worldOrient[1].item().toDouble (),
			worldOrient[2].item().toDouble ());
	pos3d << nextStep[0].item ().toDouble (), nextStep[1].item ().toDouble (), 0;
	worldFrameTot = quat * pos3d + Eigen::Vector3d(worldPos[0].item ().toDouble (), worldPos[1].item ().toDouble (),0);

	worldPoseMsg.x = worldFrameTot[0];
	worldPoseMsg.y = worldFrameTot[1];

	/************************************
	 * FINE TACCONE INCREDIBILE
	 * *********************************/
	double corridorDistance = napvig->getDistanceFromCorridor ();
	std_msgs::Float64 corridorDistanceMsg;

	corridorDistanceMsg.data = corridorDistance;

	publish ("setpoint_pub", poseMsg);
	publish ("world_setpoint_pub", worldPoseMsg);
	publish ("corridor_distance_pub", corridorDistanceMsg);
}

void NapvigNode::publishHistory ()
{
	if (nodeParams.drawWhat == TEST_DRAW_NONE )
		return;

	Napvig::SearchHistory searchHistory = napvig->getSearchHistory ();
	napvig::SearchHistory searchHistoryMsg;

	for (Tensor currPath : searchHistory.triedPaths) {
		nav_msgs::Path pathMsg;
		const int currTrials = currPath.size (0);

		pathMsg.poses.resize (currTrials);
		for (int i = 0; i < currTrials; i++) {
			pathMsg.poses[i].pose.position.x = currPath[i][0].item ().toDouble ();
			pathMsg.poses[i].pose.position.y = currPath[i][1].item ().toDouble ();
		}

		searchHistoryMsg.triedPaths.push_back (pathMsg);
	}

	for (Tensor currSearch : searchHistory.initialSearches) {
		geometry_msgs::Vector3 searchMsg;

		searchMsg.x = currSearch[0].item ().toDouble ();
		searchMsg.y = currSearch[1].item ().toDouble ();

		searchHistoryMsg.initialSearch.push_back (searchMsg);
	}

	searchHistoryMsg.chosen = searchHistory.chosen;

	publish ("search_history_pub", searchHistoryMsg);
}

void NapvigNode::publishDebug ()
{
	std_msgs::Float64MultiArray msg;

	msg.layout.dim.resize (1);
	msg.layout.dim[0].size = debug.values.size ();

	msg.data = debug.values;

	publish ("grad_log_pub", msg);
}

void NapvigNode::syncActions () {
	double val;
	PROFILE (val,[&]{
	publishValues ();
	publishControl ();
	publishHistory ();
	publishDebug ();
	});
}

int NapvigNode::actions ()
{

	return 0;
}

int main (int argc, char *argv[])
{
	init (argc, argv, NODE_NAME);

	NapvigNode npn;

	return npn.spin ();
}
