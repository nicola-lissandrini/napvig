#include "policy.h"

using namespace std;
using namespace torch;

Policy::Policy(const shared_ptr<Landscape> &_landscape):
	landscape(_landscape)
{
	flags.add ("no_trajectory");
}

Tensor SearchStraightPolicy::getNextSearch (const Napvig::Trajectory &trajectory)
{
	// Go straight with napvig output
	return trajectory.back ().search;
}

CollisionTerminatedPolicy::CollisionTerminatedPolicy (const CollisionTerminatedPolicy::Params &_params):
	params(_params)
{}

Policy::Termination CollisionTerminatedPolicy::terminationCondition (const Napvig::Trajectory &trajectory)
{
	Napvig::State last = trajectory.back ();

	if (landscape->collides (last.position))
		return PREDICTION_TERMINATION_COLLISION;

	if (trajectory.size () >= windowLength)
		return PREDICTION_TERMINATION_MAX_STEP;

	return PREDICTION_TERMINATION_NONE;
}

StartDrivenPolicy::StartDrivenPolicy(int _windowLength):
	CollisionTerminatedPolicy(_windowLength)
{}
