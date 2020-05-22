#include "napvig_node.h"

#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Pose2D.h>
#include <nav_msgs/Path.h>

using namespace ros;
using namespace std;
using namespace XmlRpc;
using namespace torch;
using namespace at;
using namespace torch::indexing;

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
	NapvigMapParams mapParams;
	NapvigParams napvigParams;

	mapParams.measureRadius = paramDouble (params["map"], "measure_radius");
	mapParams.smoothRadius = paramDouble (params["map"], "smooth_radius");
	mapParams.precision = paramInt (params["map"],"precision");

	napvigParams.stepAheadSize = paramDouble (params["napvig"], "step_ahead_size");
	napvigParams.gradientStepSize = paramDouble (params["napvig"], "gradient_step_size");
	napvigParams.terminationDistance = paramDouble (params["napvig"], "termination_distance");
	napvigParams.terminationCount = paramInt (params["napvig"], "termination_count");
	napvigParams.lookaheadHorizon = paramInt  (params["napvig"], "lookahead_horizon");
	napvigParams.algorithm = paramEnum<NapvigAlgorithmType> (params["napvig"], "algorithm", {"single_step", "predict_collision"});
	napvigParams.minDistance = paramDouble (params["napvig"],"min_distance");
	napvigParams.scatterVariance = paramDouble (params["napvig"], "scatter_variance");

	nodeParams.mapTestRangeMin = paramDouble (params["map_test"], "range_min");
	nodeParams.mapTestRangeMax = paramDouble (params["map_test"], "range_max");
	nodeParams.mapTestRangeStep = paramDouble (params["map_test"], "range_step");
	nodeParams.drawWhat = paramEnum<TestDraw> (params["map_test"],"draw", {"none","value","grad"});

	napvig = new Napvig (mapParams, napvigParams);
}

void NapvigNode::initROS () {
	addSub ("measures_sub", paramString (params,"scan_topic"), 1, &NapvigNode::measuresCallback);
	addSub ("odom_sub", paramString (params, "odom_topic"), 1, &NapvigNode::odomCallback);
	addPub<std_msgs::Float64MultiArray> ("measures_pub", paramString(params,"measures_pub_topic"),1);
	addPub<std_msgs::Float64MultiArray> ("map_values_pub", paramString(params,"map_values_pub_topic"),1);
	addPub<geometry_msgs::Pose2D> ("setpoint_pub", paramString(params,"setpoint_pub_topic"), 1);
	addPub<std_msgs::Float64MultiArray> ("grad_log_pub","/grad_log", 1);
	addPub<nav_msgs::Path> ("search_history_pub", paramString (params,"search_history_pub"), 1);
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

void NapvigNode::odomCallback (const nav_msgs::Odometry &odomMsg)
{
	complexd newFrame;
	const double realPart = odomMsg.pose.pose.orientation.w;
	const double imagPart = odomMsg.pose.pose.orientation.z;

	newFrame = pow (realPart + 1i * imagPart, 2);	// Squared since the quaternion holds half angle

	napvig->updateFrame (newFrame);
}

void NapvigNode::publishControl ()
{
	if (!napvig->isReady ())
		return;

	geometry_msgs::Pose2D poseMsg;
	torch::Tensor nextStep, nextBearing;

	napvig->resetState ();
	napvig->step ();

	nextStep = napvig->getPosition ();
	nextBearing = napvig->getBearing ();

	//cout << nextStep << endl;

	poseMsg.x = nextStep[0].item ().toDouble ();
	poseMsg.y = nextStep[1].item ().toDouble ();
	poseMsg.theta = atan2 (poseMsg.y, poseMsg.x);

	publish ("setpoint_pub", poseMsg);// temp  debug

#ifdef GRAD_DEBUG
	std_msgs::Float64MultiArray gradMsg;
	gradMsg.layout.dim.resize (2);
	gradMsg.layout.dim[0].size = napvig->gradLog.size(0);
	gradMsg.layout.dim[1].size = napvig->gradLog.size(1);
	gradMsg.data.resize (napvig->gradLog.numel ());
	const double *dataPt = (double *) napvig->gradLog.toType (ScalarType::Double).data_ptr ();
	gradMsg.data = vector<double> (dataPt, dataPt + napvig->gradLog.numel ());
	publish ("grad_log_pub", gradMsg);
#endif
}

int NapvigNode::actions ()
{
	publishValues ();
	publishControl ();

	return 0;
}

int main (int argc, char *argv[])
{
	init (argc, argv, NODE_NAME);

	NapvigNode npn;

	return npn.spin ();
}
