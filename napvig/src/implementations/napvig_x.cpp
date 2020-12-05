#include "napvig_x.h"

using namespace std;
using namespace torch;

TargetTracker::TargetTracker() {
	flags.addFlag ("first_target");
}

void TargetTracker::updateTarget(const Frame &_targetFrame) {
	flags.set("first_target");
	targetFrame = _targetFrame;
}

bool TargetTracker::targetSet() const {
	return flags["first_target"];
}

Tensor saturate (const Tensor &x, double maxValue) {
	Tensor saturated = (x > maxValue);

	return saturated * maxValue + (~saturated) * x;
}

NapvigX::NapvigX (const std::shared_ptr<Landscape::Params> &landscapeParams,
				  const std::shared_ptr<NapvigX::Params> &params):
	NapvigPredictive(NAPVIG_X,
					 landscapeParams,
					 params)
{
	fextPolicy = make_shared<FullyExploitative> (shared_ptr<Landscape> (&landscape),
												 params,
												 shared_ptr<Frame> (&targetFrame));
	/*fexpPolicy = make_shared<FullyExplorative> ();
	   pextPolicy = make_shared<PartiallyExploitative> ();*/
}

boost::optional<Napvig::Trajectory> NapvigX::trajectoryAlgorithm (const State &initialState)
{
	if (!targetSet ())
		return boost::none;

	// Check an existing Fully Exploitative trajectory exists
	return followPolicy (initialState, fextPolicy);
}

FullyExploitative::FullyExploitative(const std::shared_ptr<Landscape> &_landscape,
									 const std::shared_ptr<NapvigPredictive::Params> &_params,
									 const std::shared_ptr<Frame> &_targetFrame):
	Policy(_landscape, _params),
	targetFrame(_targetFrame)
{}

pair<Tensor,boost::optional<Tensor>> FullyExploitative::searchTowardsTarget(const Tensor &currentPosition)
{
	torch::Tensor diff = (targetFrame->position - currentPosition);
	cout << "diff " << diff.norm().item () << " rapp " << (diff.norm()  / params().stepAheadSize).item() << endl;
	return {diff / diff.norm (),
				saturate (diff.norm () / params().stepAheadSize, params().stepGainSaturation)};
}

bool FullyExploitative::checkTargetReached () const {
	return targetFrame->position.norm ().item().toDouble () < params().targetReachedThreshold;
}

pair<Tensor,boost::optional<Tensor>> FullyExploitative::getFirstSearch(const Napvig::State &initialState) {
	return searchTowardsTarget (initialState.position);
}

pair<Tensor,boost::optional<Tensor>> FullyExploitative::getNextSearch(const Napvig::Trajectory &trajectory) {
	return searchTowardsTarget (trajectory.back ().position);
}

Policy::Termination FullyExploitative::terminationCondition (const Napvig::Trajectory &trajectory)
{
	Termination collisionTerminated = CollisionTerminatedPolicy::terminationCondition (trajectory);

	if (collisionTerminated == PREDICTION_TERMINATION_NONE) {
		if (checkTargetReached ())
			return PREDICTION_TERMINATION_TARGET_REACHED;
		return PREDICTION_TERMINATION_NONE;
	}

	return collisionTerminated;
}

bool FullyExploitative::processTrajectory (const Napvig::Trajectory &trajectory, PolicyAbstract::Termination termination)
{
	index = 0;

	switch (termination) {
	case PREDICTION_TERMINATION_TARGET_REACHED:
		ROS_ERROR ("TARGET RICED");
		finalTrajectory = boost::none;
		break;
	case PREDICTION_TERMINATION_MAX_STEP:
		ROS_INFO ("NON HOVVISTO TUTTO");
		finalTrajectory = trajectory;
		break;
	case PREDICTION_TERMINATION_COLLISION:
		ROS_WARN ("COLLIDEEEEEE NO TRAJ");
		finalTrajectory = boost::none;
		break;
	default:
		cout << "unexp " << termination << endl;
		break;
	}



	// Fully-Explorative is one-shot, the trajectory either exists or it doesn't
	// So always stop
	return false;
}








