#ifndef NAPVIG_NODE_H
#define NAPVIG_NODE_H

#include "sparcsnode.h"
#include "napvig.h"
#include <torch/torch.h>
#include <ATen/Tensor.h>

#include <sensor_msgs/LaserScan.h>

#define NODE_NAME "napvig"

enum TestDraw {
	TEST_DRAW_NONE = 0,
	TEST_DRAW_VALUE,
	TEST_DRAW_GRAD
};

struct NapvigNodeParams {
	double mapTestRangeMin;
	double mapTestRangeMax;
	double mapTestRangeStep;
	TestDraw drawWhat;
};

class NapvigNode : public SparcsNode
{
	Napvig *napvig;
	NapvigNodeParams nodeParams;
	struct {
		torch::Tensor points;
		int xySize;
	} testGrid;

	void initTestGrid ();

	void initParams ();
	void initROS ();
	int actions ();

	torch::Tensor convertScanMsg (const sensor_msgs::LaserScan &scanMsg);

	void publishMeasures(const torch::Tensor &measures);
	void publishValues ();
	void publishControl ();

	void measuresCallback(const sensor_msgs::LaserScan &scanMsg);

public:
	NapvigNode ();
};

#endif // NAPVIG_NODE_H
