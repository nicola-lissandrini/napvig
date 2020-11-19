#include "napvig_node.h"
#include "multi_array_manager.h"

using namespace std;
using namespace ros;
using namespace XmlRpc;
using namespace torch;
using namespace std::placeholders;

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
	SparcsNode(NODE_NAME),
	napvigHandler(std::bind(
					  &NapvigNode::publishControl,
					  this,
					  std::placeholders::_1))
{
	initParams ();
	initROS ();
}

void NapvigNode::initParams ()
{
	vector<string> algorithmList = {
		"napvig_legacy",
		"napvig_randomized",
		"napvig_x",
		"napvig_cube",
		"napvig_hyper_dim",
		"napvig_fusion",
		"napvig_collaborative"
	};
	Napvig::AlgorithmType algorithm = paramEnum<Napvig::AlgorithmType> (params, "algorithm", algorithmList);
	napvigHandler.init (algorithm, params);
	napvigDebugger = make_shared<NapvigNodeDebugger> (napvigHandler.getDebug (), params["debug"]);
}

void NapvigNode::initCoreTopics (XmlRpcValue &params) {
	addSub ("measures_sub", paramString (params["subs"],"scan"), 1, &NapvigNode::measuresCallback);
	addSub ("odom_sub", paramString (params["subs"], "odom"), 5, &NapvigNode::odomCallback);

	addPub<geometry_msgs::Pose2D> ("command_pub", paramString(params["pubs"],"command"), 1);
}

void NapvigNode::initDebugTopics (XmlRpcValue &params) {
	addPub<std_msgs::Float32MultiArray> ("measures_pub", paramString(params["pubs"],"measures"),1);
	addPub<std_msgs::Float32MultiArray> ("values_pub", paramString(params["pubs"],"landscape_values"),1);
	addPub<std_msgs::Float32MultiArray> ("debug_pub",paramString(params["pubs"],"generic_vector"), 1);
	addPub<napvig::SearchHistory> ("search_history_pub", paramString (params["pubs"],"search_history"), 1);
}

void NapvigNode::initNapvigXTopics (XmlRpcValue &params) {
	addSub ("target_sub", paramString (params["subs"],"target"), 1, &NapvigNode::targetCallback);
}

void NapvigNode::initROS ()
{
	initCoreTopics (params["topics"]["napvig_core"]);
	initDebugTopics (params["topics"]["debug"]);

	switch (napvigHandler.getType ()) {
	case Napvig::NAPVIG_LEGACY:
		// Nothing to add
		break;
	case Napvig::NAPVIG_X:
		initNapvigXTopics (params["topics"]["napvig_x"]);
		break;
	default:
		break;
	}
}

void NapvigNode::publishDebug () const
{
	if (napvigDebugger->checkPublishValues ()) {
		std_msgs::Float32MultiArray msg;
		napvigDebugger->buildValuesMsg (msg);
		publish("values_pub", msg);
	}
	if (napvigDebugger->checkPublishHistory ()) {
		napvig::SearchHistory msg;
		napvigDebugger->buildHistoryMsg (msg);
		publish("search_history_pub", msg);
	}
	if (napvigDebugger->checkPublishDebug ()) {
		std_msgs::Float32MultiArray msg;
		napvigDebugger->buildDebugMsg (msg);
		publish("debug_pub", msg);
	}
}

void NapvigNode::publishControl (const torch::Tensor &command) const
{
	geometry_msgs::Pose2D commandMsg;

	commandMsg.x = command[0].item ().toDouble ();
	commandMsg.y = command[1].item ().toDouble ();

	publish ("command_pub", commandMsg);

	publishDebug ();
}

void NapvigNode::measuresCallback (const sensor_msgs::LaserScan &scanMsg)
{
	auto measures = make_shared<LidarMeasures> (scanMsg.ranges,
												scanMsg.angle_min,
												scanMsg.angle_increment);
	napvigHandler.updateMeasures (measures);

	if (napvigDebugger->checkPublishMeasures ()) {
		std_msgs::Float32MultiArray msg;
		napvigDebugger->buildMeasuresMsg (measures, msg);
		publish ("measures_pub", msg);
	}
}

void NapvigNode::odomCallback (const nav_msgs::Odometry &odomMsg)
{
	torch::Tensor quaternion = quaternionMsgToTorch (odomMsg.pose.pose.orientation);
	torch::Tensor position = pointMsgToTorch (odomMsg.pose.pose.position).slice (0,0,2);
	napvigHandler.updateFrame (Frame{Rotation(quaternion), position});
}

void NapvigNode::targetCallback (const geometry_msgs::Pose &targetMsg) {
	assert (false && "Target callback not implemented");
}

int NapvigNode::actions () {
	return napvigHandler.synchronousActions ();
}

int main (int argc, char *argv[])
{
	init (argc, argv, NODE_NAME);

	NapvigNode npn;

	return npn.spin ();
}






















