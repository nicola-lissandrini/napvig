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

pair<Napvig::Trajectory, Policy::Termination> NapvigPredictive::predictTrajectory (const Napvig::State &initialState, const shared_ptr<Policy> &policy)
{
	Napvig::Trajectory predicted;
	Napvig::State current;
	Policy::Termination condition;

	current = initialState.clone ();
	// First trajectory sample is initial state
	predicted.push_back (initialState);
	QUA;
	// Iterate until current policy termination condition arises
	do {
		// Update search direction according to policy
		current.search = policy->getNextSearch (predicted); QUA;

		cout << "Search\n" << current.search << endl;
		cout << "Pos\n" << current.position << endl << endl;
		// Compute next step
		current = core.compute (current); QUA;
		// Store in trajectory
		predicted.push_back (current); QUA;

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

	// Until a trajectory is selected..
	do {
		// Get initial search according to policy
		Tensor firstSearch = policy->getFirstSearch (initialState);
		State firstState = {initialState.position, firstSearch};

		// Perform prediction
		tie (trajectory, termination) = predictTrajectory (firstState, policy);
	} while (!policy->selectTrajectory (trajectory, termination)); // Choose policy if validated or terminate if none is found

	// Check whether no suitable trajectory is found
	if (policy->noTrajectory ())
		return boost::none;
	else
		return trajectory;
}

/***********
 * Policies
 * *********/

Tensor SearchStraightPolicy::getNextSearch (const Napvig::Trajectory &trajectory)
{
	// Go straight with napvig output
	return trajectory.back ().search;
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


