#include "napvig_node.h"

#include <std_msgs/Float64MultiArray.h>
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
	napvigParams.algorithm = paramEnum<Napvig::AlgorithmType> (params["napvig"], "algorithm", {"single_step", "predict_collision"});
	napvigParams.minDistance = paramDouble (params["napvig"],"min_distance");
	napvigParams.scatterVariance = paramDouble (params["napvig"], "scatter_variance");

	nodeParams.mapTestRangeMin = paramDouble (params["map_test"], "range_min");
	nodeParams.mapTestRangeMax = paramDouble (params["map_test"], "range_max");
	nodeParams.mapTestRangeStep = paramDouble (params["map_test"], "range_step");
	nodeParams.drawWhat = paramEnum<TestDraw> (params["map_test"],"draw", {"none","value","grad"});

	napvig = new Napvig (mapParams, napvigParams);
}

void NapvigNode::initROS ()
{
	addSub ("measures_sub", paramString (params["topics"]["subs"],"scan"), 1, &NapvigNode::measuresCallback);
	addSub ("odom_sub", paramString (params["topics"]["subs"], "odom"), 1, &NapvigNode::odomCallback);
	addSub ("target_sub", paramString (params["topics"]["subs"], "target"), 1, &NapvigNode::targetCallback);

	addPub<std_msgs::Float64MultiArray> ("measures_pub", paramString(params["topics"]["pubs"],"measures"),1);
	addPub<std_msgs::Float64MultiArray> ("map_values_pub", paramString(params["topics"]["pubs"],"map_values"),1);
	addPub<geometry_msgs::Pose2D> ("setpoint_pub", paramString(params["topics"]["pubs"],"setpoint"), 1);
	addPub<std_msgs::Float64MultiArray> ("grad_log_pub","/grad_log", 1);
	addPub<napvig::SearchHistory> ("search_history_pub", paramString (params["topics"]["pubs"],"search_history"), 1);
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
	if (!napvig->isMapReady () || nodeParams.drawWhat == TEST_DRAW_NONE)
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

void NapvigNode::measuresCallback (const sensor_msgs::LaserScan &scanMsg)
{
	torch::Tensor newMeasures = convertScanMsg (scanMsg);

	napvig->setMeasures (newMeasures);
	publishMeasures (newMeasures);

}

void NapvigNode::odomCallback (const nav_msgs::Odometry &odomMsg) {
	napvig->updateFrame (Frame{Rotation (quaternionMsgToTorch (odomMsg.pose.pose.orientation)),
							   pointMsgToTorch (odomMsg.pose.pose.position).slice (0,0,2)});
}

void NapvigNode::targetCallback (const geometry_msgs::Pose &targetMsg) {
	napvig->updateTarget (Frame{Rotation (quaternionMsgToTorch (targetMsg.orientation)),
								pointMsgToTorch (targetMsg.position).slice (0,0,2)});
}

void NapvigNode::publishControl ()
{
	if (!napvig->isReady ())
		return;
	geometry_msgs::Pose2D poseMsg;
	torch::Tensor nextStep, nextBearing;

	if (!napvig->step ())
		return;

	nextStep = napvig->getSetpointPosition ();
	nextBearing = napvig->getSetpointDirection ();

	poseMsg.x = nextStep[0].item ().toDouble ();
	poseMsg.y = nextStep[1].item ().toDouble ();
	poseMsg.theta = atan2 (poseMsg.y, poseMsg.x);

	publish ("setpoint_pub", poseMsg);
}



void NapvigNode::publishHistory ()
{
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

	publish ("search_history_pub", searchHistoryMsg);
}

int NapvigNode::actions ()
{
	publishValues ();
	publishControl ();
	publishHistory ();

	return 0;
}

int main (int argc, char *argv[])
{
	init (argc, argv, NODE_NAME);

	NapvigNode npn;

	return npn.spin ();
}
