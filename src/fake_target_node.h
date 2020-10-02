#ifndef FAKE_TARGET_NODE_H
#define FAKE_TARGET_NODE_H

#include "sparcsnode.h"
#include <tf/tf.h>

#include <nav_msgs/Odometry.h>

#define NODE_NAME "fake_target"

class FakeTargetNode : public SparcsNode
{
	tf::Pose bodyFrame, targetFrame;

	void initParams ();
	void initROS ();
	int actions ();

	void publishTargetInBody (const tf::Pose &targetInBodyFrame);
	tf::Pose targetInBody (const tf::Pose &targetFrame, const tf::Pose &bodyFrame);

public:
	FakeTargetNode ();

	void odomCallback (const nav_msgs::Odometry &odomMsg);
	void poseCallback (const geometry_msgs::Pose &poseMsg);
};

#endif // FAKE_TARGET_NODE_H
