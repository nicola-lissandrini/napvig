#ifndef NAPVIG_NODE_H
#define NAPVIG_NODE_H

#include "sparcsnode.h"
#include "napvig.h"
#include <torch/torch.h>
#include <ATen/Tensor.h>

#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>

#define NODE_NAME "napvig"

torch::Tensor quaternionMsgToTorch (const geometry_msgs::Quaternion &quaternionMsg);

enum TestDraw {
	TEST_DRAW_NONE = 0,
	TEST_DRAW_VALUE,
	TEST_DRAW_GRAD,
	TEST_DRAW_MINIMAL
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
	NapvigDebug debug;
	NapvigNodeParams nodeParams;
	torch::Tensor worldPos, worldOrient;

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
	void publishHistory();
	void publishDebug();

	void measuresCallback(const sensor_msgs::LaserScan &scanMsg);
	void odomCallback (const nav_msgs::Odometry &odomMsg);
	void targetCallback (const geometry_msgs::Pose &targetMsg);
	void corridorCallback(const std_msgs::Float32MultiArray &corridorMsg);

	void syncActions();
public:
	NapvigNode ();
};

#endif // NAPVIG_NODE_H
