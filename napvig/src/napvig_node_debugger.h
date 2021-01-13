#ifndef NAPVIG_NODE_DEBUGGER_H
#define NAPVIG_NODE_DEBUGGER_H

#include "napvig.h"
#include "napvig_handler.h"
#include "napvig/SearchHistory.h"
#include <std_msgs/Float32MultiArray.h>
#include "multi_array_manager.h"


enum TestDraw {
	TEST_DRAW_NONE = 0,
	TEST_DRAW_VALUE,
	TEST_DRAW_GRAD,
	TEST_DRAW_LANDMARKS
};

class NapvigNodeDebugger
{
	struct Params {
		double mapTestRangeMin;
		double mapTestRangeMax;
		double mapTestRangeStep;
		bool publishMeasures;
		bool publishHistory;
		bool worldFrameView;
		TestDraw drawWhat;
	} params;

	NapvigDebug::Ptr debug;

	struct {
		torch::Tensor points;
		int xySize;
	} testGrid;

	void initTestGrid ();
	void toWorldFrame(torch::Tensor &measures) const;

	void initParams (XmlRpc::XmlRpcValue &_params);
	void valuesFromValues (std_msgs::Float32MultiArray &valuesMsg) const;
	void valuesFromGrad (std_msgs::Float32MultiArray &valuesMsg) const;
	void valuesFromLandmarks (std_msgs::Float32MultiArray &valuesMsg) const;

public:
	NapvigNodeDebugger (const std::shared_ptr<NapvigDebug> &_debug, XmlRpc::XmlRpcValue &_params);

	void buildMeasuresMsg (const std::shared_ptr<Measures> &measures, std_msgs::Float32MultiArray &measuresMsg) const;
	void buildHistoryMsg (napvig::SearchHistory &searchHistoryMsg) const;
	void buildDebugMsg (std_msgs::Float32MultiArray &debugMsg) const;
	void buildValuesMsg (std_msgs::Float32MultiArray &valuesMsg) const;

	bool checkPublishMeasures () const;
	bool checkPublishHistory () const;
	bool checkPublishDebug ()  const;
	bool checkPublishValues () const;

	DEF_SHARED (NapvigNodeDebugger);
};

#endif // NAPVIG_NODE_DEBUGGER_H
