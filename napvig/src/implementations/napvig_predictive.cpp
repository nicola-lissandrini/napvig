#include "napvig_predictive.h"

using namespace std;
using namespace torch;

NapvigPredictive::NapvigPredictive (AlgorithmType type,
									const Landscape::Params &landscapeParams,
									const NapvigPredictive::Params &predictiveParams):
	Napvig (type,
			landscapeParams,
			predictiveParams.napvigParams),
	params(predictiveParams)
{
	flags.addFlag ("first_policy_set", true);
}

void NapvigPredictive::setPolicy(std::shared_ptr<Policy> _policy)
{
	policy = _policy;
	flags.set("first_policy");
}

pair<Napvig::Trajectory, Policy::Termination> NapvigPredictive::predictTrajectory (const Napvig::State &initialState, const shared_ptr<Policy> &policy)
{
	Napvig::Trajectory predicted;
	Napvig::State current;
	Policy::Termination condition;
	
	// First trajectory sample is initial state
	predicted.push_back (initialState);

	// Iterate until current policy termination condition arises
	do {
		// Update search direction according to policy
		current.search = policy->getNextSearch (predicted);
		// Compute next step
		current = core.compute (current);
		// Store in trajectory
		predicted.push_back (current);

		condition = policy->terminationCondition (predicted);
	} while (condition == Policy::PREDICTION_TERMINATION_NONE);

	return {predicted, condition};
}

boost::optional<Napvig::Trajectory> NapvigPredictive::trajectoryAlgorithm (const State &initialState)
{
	if (!isReady ())
		return boost::none;

	Napvig::Trajectory trajectory;
	Policy::Termination termination;

	// Initialize policy if needed
	policy->init ();

	// Until a trajectory is selected
	do {
		// Get initial search according to policy
		Tensor firstSearch = policy->getFirstSearch (initialState);
		State firstState = {initialState.position, firstSearch};

		// Perform prediction
		tie (trajectory, termination) = predictTrajectory (firstState, policy);
	} while (!selectTrajectory (trajectory, termination)); // Choose policy if validated or terminate if none is found

	// Check whether no suitable trajectory is found
	if (policy->noTrajectory ())
		return boost::none;
	else
		return trajectory;
}


