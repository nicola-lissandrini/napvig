#include "unicycle_control.h"

#include <Eigen/Geometry>
#include <unsupported/Eigen/EulerAngles>

#include <complex>

using namespace ros;
using namespace std;
using namespace XmlRpc;
using namespace Eigen;

typedef complex<double> complexd;

UnicycleControl::UnicycleControl():
	Controller ()
{
}

UnicycleControlNode::UnicycleControlNode ():
	SparcsControlNode(NODE_NAME, &uc),
	uc()
{
	initParams ();

	start ();
}

#define SETPOINT_SOURCE 1

void UnicycleControl::updateInput (const VectorXd &state, const VectorXd &ref)
{
	const double kOmega = params["gains"][0];
	const double kVMax = params["gains"][1];
	const double scaleFactor = params["gains"][2];
	const double targetTheta  = ref[2];

	complexd b0 = polar (1.0, 0.0);
	complexd b1 = polar (1.0, targetTheta);
	complexd diff = log (b1 / b0);
	const double omega = kOmega * imag (diff);

	control[0] = kVMax * exp (-pow(omega,2)/(2* pow (M_PI * scaleFactor,2)));
	control[1] = omega;
}

void UnicycleControlNode::initControl ()
{
	ControlParams ctrlParams;

	ctrlParams.refSize = D_3D;
	ctrlParams.stateSize = D_2D + D_2D; // x y th_w th_z
	ctrlParams.controlSize = D_2D;

	ctrlParams.sampleTime = 1 / rate->expectedCycleTime ().toSec ();

	ctrlParams.params = {
		{"gains", paramVector (params,"gains")}
	};

	uc.setParams (ctrlParams);
}

void UnicycleControlNode::initParams () {
	initControl ();
}

void UnicycleControlNode::stateConvertMsg (VectorXd &_state, const nav_msgs::Odometry &_stateMsg)
{
	_state << _stateMsg.pose.pose.position.x,
			_stateMsg.pose.pose.position.y,
			_stateMsg.pose.pose.orientation.w,
			_stateMsg.pose.pose.orientation.z;

}

void UnicycleControlNode::refConvertMsg (VectorXd &_ref, const geometry_msgs::Pose2D &_refMsg)
{
	_ref << _refMsg.x,
			_refMsg.y,
			_refMsg.theta;
}


void UnicycleControlNode::commandConvertMsg (geometry_msgs::Twist &_commandMsg, const VectorXd &_command)
{
	_commandMsg.linear.x = _command[0];
	_commandMsg.angular.z = _command[1];
}

int main (int argc, char *argv[])
{
	init (argc, argv, NODE_NAME);
	UnicycleControlNode ucn;

	return ucn.spin ();
}


