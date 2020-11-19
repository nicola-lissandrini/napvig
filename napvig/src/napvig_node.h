#ifndef NAPVIG_NODE_H
#define NAPVIG_NODE_H

#include "sparcsnode.h"
#include "napvig_handler.h"
#include <torch/torch.h>
#include <ATen/Tensor.h>

#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>

#include <geometry_msgs/Pose2D.h>
#include <napvig/SearchHistory.h>
#include "napvig_node_debugger.h"

#define NODE_NAME "napvig"

torch::Tensor quaternionMsgToTorch (const geometry_msgs::Quaternion &quaternionMsg);
torch::Tensor pointMsgToTorch (const geometry_msgs::Point &vectorMsg);

class NapvigNode : public SparcsNode
{
	NapvigHandler napvigHandler;
	std::shared_ptr<NapvigNodeDebugger> napvigDebugger;

	void initCoreTopics(XmlRpc::XmlRpcValue &params);
	void initDebugTopics(XmlRpc::XmlRpcValue &params);

	// Implementation dependant topics
	void initNapvigXTopics (XmlRpc::XmlRpcValue &params);

	void initParams ();
	void initROS ();
	int actions ();

	void publishControl (const torch::Tensor &command) const;
	void publishDebug () const;


	void measuresCallback(const sensor_msgs::LaserScan &scanMsg);
	void odomCallback (const nav_msgs::Odometry &odomMsg);
	void targetCallback (const geometry_msgs::Pose &targetMsg);

public:
	NapvigNode ();
};

#endif // NAPVIG_NODE_H
