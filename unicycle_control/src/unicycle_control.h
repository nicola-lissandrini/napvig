#ifndef UNICYCLE_CONTROL_H
#define UNICYCLE_CONTROL_H

#define DISABLE_PROFILE_OUTPUT
#define NODE_NAME "unicycle_control"
#include "controller_node.h"

#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Empty.h>

class UnicycleControl : public Controller
{
	int t;
	double diffOld;
	double derivLock;
	double diffIntegr;

	double updateDeriv (double diff);
	double updateIntegr (double diff);

public:
	UnicycleControl ();

	void setPid (const ControlParams &params);

	void updateInput (const Eigen::VectorXd &state, const Eigen::VectorXd &ref = Eigen::VectorXd ());
};

class UnicycleControlNode : public SparcsControlNode
		<nav_msgs::Odometry,
		 geometry_msgs::Pose2D,
		 geometry_msgs::Twist,
		 true>
{
	UnicycleControl uc;

	void initControl ();
	void initParams ();

	void stateConvertMsg (Eigen::VectorXd &_state, const nav_msgs::Odometry &_stateMsg);
	void refConvertMsg (Eigen::VectorXd &_ref, const geometry_msgs::Pose2D &_refMsg);
	void commandConvertMsg (geometry_msgs::Twist &_commandMsg, const Eigen::VectorXd &_command);

public:
	UnicycleControlNode ();
};

#endif // UNICYCLE_CONTROL_H
