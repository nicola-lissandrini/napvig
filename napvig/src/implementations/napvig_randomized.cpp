#include "napvig_randomized.h"

using namespace std;
using namespace torch;

NapvigRandomized::NapvigRandomized (const Landscape::Params::Ptr &_landscapeParams,
									const NapvigRandomized::Params::Ptr &_params):
	NapvigPredictive(NAPVIG_RANDOMIZED,
					 _landscapeParams,
					 _params)
{
	randomizePolicy = make_shared <RandomizePolicy> (shared_ptr<Landscape> (&landscape), _params);
}

boost::optional<Napvig::Trajectory> NapvigRandomized::trajectoryAlgorithm (const Napvig::State &initialState) {
	return followPolicy (initialState, randomizePolicy);
}

RandomizePolicy::RandomizePolicy (const Landscape::Ptr _landscape,
								  const NapvigRandomized::Params::Ptr &_params):
	Policy(_landscape, _params),
	first(false)
{
}

void RandomizePolicy::init ()
{
	trials = 0;
	index = 0;
	first = true;
}

Tensor RandomizePolicy::randomize (const torch::Tensor &search)
{
	double thetaRandom = torch::normal (0.0, params().randomizeVariance, {1}).item ().toDouble ();
	Rotation randomizedRotation = Rotation::fromAxisAngle (thetaRandom);

	return randomizedRotation * search;
}

pair<Tensor,boost::optional<Tensor>> RandomizePolicy::getFirstSearch (const Napvig::State &initialState)
{
	if (first) {
		first = false;
		lastSearch = initialState.search;

		return {lastSearch, boost::none};
	}  else {
		lastSearch = randomize (lastSearch);

		return {lastSearch, boost::none};
	}
}

bool RandomizePolicy::processTrajectory (const Napvig::Trajectory &trajectory, Termination termination)
{
	bool trialsExceeded = trials >= params().maxTrials;

	if (termination == PREDICTION_TERMINATION_MAX_STEP && !trialsExceeded) {
		finalTrajectoryIndexed = {trajectory, trials};

		return false;
	}

	if (trialsExceeded) {
		finalTrajectoryIndexed = {boost::none, -1};

		return false;
	}

	trials++;

	return true;
}
