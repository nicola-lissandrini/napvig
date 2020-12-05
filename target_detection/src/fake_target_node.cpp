#include "fake_target_node.h"

using namespace ros;
using namespace std;
using namespace tf;

FakeTargetNode::FakeTargetNode ():
	SparcsNode(NODE_NAME)
{
	initParams ();
	initROS ();

	targetFrame.setIdentity ();
}

void FakeTargetNode::initParams ()
{
}

void FakeTargetNode::publishTargetInBody (const Pose &targetInBodyFrame)
{
	geometry_msgs::Pose targetInBodyMsg;

	poseTFToMsg (targetInBodyFrame, targetInBodyMsg);
	publish("body_frame_target_pub", targetInBodyMsg);
}

Pose FakeTargetNode::targetInBody (const Pose &targetFrame, const Pose &bodyFrame) {
	return bodyFrame.inverse () * targetFrame;
}

int FakeTargetNode::actions ()
{
	Pose targetInBodyFrame = targetInBody (targetFrame, bodyFrame);

	publishTargetInBody (targetInBodyFrame);

	return 0;
}

void FakeTargetNode::initROS ()
{
	addSub ("odom_sub", paramString (params, "odom_sub"), 1, &FakeTargetNode::odomCallback);
	addSub ("target_pos_sub", paramString (params, "target_pos_sub"), 1, &FakeTargetNode::poseCallback);

	addPub<geometry_msgs::Pose> ("body_frame_target_pub", paramString (params, "body_frame_target_pub"), 1);
}

void FakeTargetNode::odomCallback (const nav_msgs::Odometry &odomMsg) {
	poseMsgToTF (odomMsg.pose.pose, bodyFrame);
}

void FakeTargetNode::poseCallback (const geometry_msgs::Pose &poseMsg) {
	poseMsgToTF (poseMsg, targetFrame);
}

int main (int argc, char *argv[])
{
	init (argc, argv, NODE_NAME);

	FakeTargetNode ftn;

	return ftn.spin ();
}
























