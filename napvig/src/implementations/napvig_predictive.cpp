#include "napvig_predictive.h"

using namespace std;
using namespace torch;

NapvigPredictive::NapvigPredictive (AlgorithmType type,
									const shared_ptr<Landscape::Params> &landscapeParams,
									const shared_ptr<Napvig::Params> &napvigParams):
	Napvig (type,
			landscapeParams,
			napvigParams)
{
}

pair<Napvig::Trajectory, Policy::Termination> NapvigPredictive::predictTrajectory (const State &initialState, const shared_ptr<Policy> &policy)
{
	Napvig::Trajectory predicted;
	Napvig::State current;
	Policy::Termination condition;

	current = initialState.clone ();
	// First trajectory sample is initial state
	predicted.push_back (initialState);

	// Iterate until current policy termination condition arises
	do {
		// Update search direction according to policy
		tie (current.search, current.stepGain) = policy->getNextSearch (predicted);

		// Compute next step
		current = core.compute (current);

		// Store in trajectory
		predicted.push_back (current);

		condition = policy->terminationCondition (predicted);
	} while (condition == Policy::PREDICTION_TERMINATION_NONE);

	return {predicted, condition};
}

boost::optional<Napvig::Trajectory> NapvigPredictive::followPolicy (const State &initialState, const std::shared_ptr<Policy> &policy)
{
	Napvig::Trajectory trajectory;
	Policy::Termination termination;

	// Initialize policy if needed
	policy->init ();
	debug->history.reset ();

	// Until a trajectory is selected..
	do {
		// Get initial search according to policy
		State firstState = initialState.clone ();
		tie (firstState.search, firstState.stepGain) = policy->getFirstSearch (initialState);

		// Perform prediction
		tie (trajectory, termination) = predictTrajectory (firstState, policy);
		
		// Debug
		debug->history.add (trajectory);
	} while (policy->processTrajectory (trajectory, termination)); // Iterate until processTrajectory returns false

	// Return final trajectory
	boost::optional<Napvig::Trajectory> finalTrajectory;
	int chosen;

	tie (finalTrajectory, chosen) = policy->getFinalTrajectory ();
	debug->history.chosen = chosen;

	return finalTrajectory;
}

/***********
 * Policies
 * *********/

pair<Tensor,boost::optional<Tensor>> SearchStraightPolicy::getNextSearch (const Napvig::Trajectory &trajectory)
{
	// Go straight with napvig output
	return {trajectory.back ().search, boost::none};
}

Policy::Termination CollisionTerminatedPolicy::terminationCondition (const Napvig::Trajectory &trajectory)
{
	Napvig::State last = trajectory.back ();

	if (landscape->collides (last.position))
		return PREDICTION_TERMINATION_COLLISION;

	if (trajectory.size () >= params().windowLength)
		return PREDICTION_TERMINATION_MAX_STEP;

	return PREDICTION_TERMINATION_NONE;
}

StartDrivenPolicy::StartDrivenPolicy(const std::shared_ptr<const Landscape> &_landscape,
									 const std::shared_ptr<const NapvigPredictive::Params> &_params):
	Policy(_landscape, _params)
{}


