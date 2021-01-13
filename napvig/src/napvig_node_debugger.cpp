#include "napvig_node_debugger.h"

using namespace ros;
using namespace std;
using namespace XmlRpc;
using namespace torch;
using namespace torch::indexing;

void NapvigNodeDebugger::initParams(XmlRpcValue &_params)
{
	params.mapTestRangeMin = paramDouble (_params["landscape_test"], "range_min");
	params.mapTestRangeMax = paramDouble (_params["landscape_test"], "range_max");
	params.mapTestRangeStep = paramDouble (_params["landscape_test"], "range_step");
	params.drawWhat = paramEnum<TestDraw> (_params["landscape_test"],"draw", {"none","value","grad","landmarks"});
	params.publishMeasures = paramBool (_params,"publish_measures");
	params.publishHistory = paramBool (_params,"publish_history");
	params.worldFrameView = paramBool (_params,"world_frame_view");
}

void NapvigNodeDebugger::initTestGrid()
{
	Tensor xyRange = torch::arange (params.mapTestRangeMin, params.mapTestRangeMax, params.mapTestRangeStep, torch::dtype (kFloat64));
	Tensor xx, yy;
	vector<Tensor> xy;

	xy = meshgrid ({xyRange, xyRange});

	xx = xy[0].reshape (-1);
	yy = xy[1].reshape (-1);

	testGrid.points = stack ({xx, yy}, 1);
	testGrid.xySize = xyRange.size (0);
}

NapvigNodeDebugger::NapvigNodeDebugger (const std::shared_ptr<NapvigDebug> &_debug, XmlRpcValue &_params):
	debug(_debug)
{
	initParams (_params);
	initTestGrid ();
}

void NapvigNodeDebugger::toWorldFrame (torch::Tensor &measures) const
{
	const Frame current = debug->framesTrackerPtr->current();
	for (int i = 0; i < measures.size (0); i++)
		measures[i] = current * measures[i];
}

void NapvigNodeDebugger::buildMeasuresMsg (const std::shared_ptr<Measures> &measures, std_msgs::Float32MultiArray &measuresMsg) const
{
	torch::Tensor measuresTensor = measures->get ().to (torch::kFloat);

	if (params.worldFrameView)
		toWorldFrame (measuresTensor);

	MultiArray32Manager array({measuresTensor.size(0),
							   measuresTensor.size(1)});

	vector<float> &data = array.data ();

	data = vector<float> ((float*)measuresTensor.data_ptr (), (float*)measuresTensor.data_ptr () + measuresTensor.numel());

	measuresMsg = array.getMsg ();
}

void NapvigNodeDebugger::buildHistoryMsg (napvig::SearchHistory &searchHistoryMsg) const
{
	for (const auto &pathTrial : debug->history.trials) {
		nav_msgs::Path pathMsg;
		const int currPathLength = pathTrial.path.size (0);

		pathMsg.poses.resize (currPathLength);
		for (int i = 0; i < currPathLength; i++) {
			pathMsg.poses[i].pose.position.x = pathTrial.path[i][0].item ().toDouble ();
			pathMsg.poses[i].pose.position.y = pathTrial.path[i][1].item ().toDouble ();
		}

		geometry_msgs::Vector3 searchMsg;
		searchMsg.x = pathTrial.initialSearch[0].item ().toDouble ();
		searchMsg.y = pathTrial.initialSearch[1].item ().toDouble ();

		searchHistoryMsg.initialSearch.push_back (pathMsg);
	}

	searchHistoryMsg.chosen = debug->history.chosen;
}

void NapvigNodeDebugger::buildDebugMsg (std_msgs::Float32MultiArray &debugMsg) const
{
	MultiArray32Manager array({debug->values.size ()});

	array.data () = debug->values;

	debugMsg = array.getMsg ();
}

#define RANGE_DIM 3

void NapvigNodeDebugger::valuesFromValues (std_msgs::Float32MultiArray &valuesMsg) const
{
	MultiArray32Manager array({testGrid.xySize,
							   testGrid.xySize},
							  RANGE_DIM);

	array.data ()[0] = params.mapTestRangeMin;
	array.data ()[1] = params.mapTestRangeMax;
	array.data ()[2] = params.mapTestRangeStep;

	for (int i = 0; i < testGrid.points.size (0); i++) {
		Tensor currPoint = testGrid.points.index ({i, None});

		array.data ()[RANGE_DIM + i] = debug->landscapePtr->value (currPoint).item ().toDouble ();
	}

	valuesMsg = array.getMsg ();
}

void NapvigNodeDebugger::valuesFromLandmarks(std_msgs::Float32MultiArray &valuesMsg) const
{
	MultiArray32Manager array({testGrid.xySize,
							  testGrid.xySize}, RANGE_DIM);

	array.data ()[0] = params.mapTestRangeMin;
	array.data ()[1] = params.mapTestRangeMax;
	array.data ()[2] = params.mapTestRangeStep;

	debug->explorativeCostPtr->convertLandmarks ();

	for (int i = 0; i < testGrid.points.size (0); i++) {
		Tensor currPoint = testGrid.points.index ({i, None});

		array.data ()[RANGE_DIM + i] = debug->explorativeCostPtr->pointCost (currPoint, params.worldFrameView);
	}

	valuesMsg = array.getMsg ();
}


void NapvigNodeDebugger::valuesFromGrad (std_msgs::Float32MultiArray &valuesMsg) const
{
	MultiArray32Manager array({testGrid.xySize,
							   testGrid.xySize,
							   debug->landscapePtr->getDim ()},
							  RANGE_DIM);

	array.data ()[0] = params.mapTestRangeMin;
	array.data ()[1] = params.mapTestRangeMax;
	array.data ()[2] = params.mapTestRangeStep;

	for (int i = 0; i < testGrid.points.size (0); i++) {
		Tensor currPoint = testGrid.points.index ({i, None});
		Tensor grad = debug->landscapePtr->grad (currPoint);

		array.data()[RANGE_DIM + 2*i] = grad[0].item ().toDouble ();
		array.data()[RANGE_DIM + 2*i + 1] = grad[1].item ().toDouble ();
	}

	valuesMsg = array.getMsg ();
}
void NapvigNodeDebugger::buildValuesMsg (std_msgs::Float32MultiArray &valuesMsg) const
{
	switch (params.drawWhat) {
	case TEST_DRAW_VALUE:
		valuesFromValues (valuesMsg);
		break;
	case TEST_DRAW_GRAD:
		valuesFromGrad (valuesMsg);
		break;
	case TEST_DRAW_LANDMARKS:
		valuesFromLandmarks (valuesMsg);
	default:
		return;
	}
}

bool NapvigNodeDebugger::checkPublishMeasures()  const {
	return params.publishMeasures;
}

bool NapvigNodeDebugger::checkPublishHistory()  const {
	return (debug->landscapePtr->isReady () && params.publishHistory);
}

bool NapvigNodeDebugger::checkPublishDebug() const {
	return (debug->values.size () > 0);
}

bool NapvigNodeDebugger::checkPublishValues() const {
	if (params.drawWhat == TEST_DRAW_LANDMARKS)
		return debug->explorativeCostPtr != nullptr;
	return (debug->landscapePtr->isReady () && (params.drawWhat == TEST_DRAW_VALUE ||
											 params.drawWhat == TEST_DRAW_GRAD));
}

























