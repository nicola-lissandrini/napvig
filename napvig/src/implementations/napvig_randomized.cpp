#include "napvig_randomized.h"

using namespace std;
using namespace torch;

NapvigRandomized::NapvigRandomized (const shared_ptr<Landscape::Params> &_landscapeParams,
									const shared_ptr<NapvigRandomized::Params> &_params):
	NapvigPredictive(NAPVIG_RANDOMIZED,
					 _landscapeParams,
					 _params)
{
	randomizePolicy = make_shared <RandomizePolicy> (shared_ptr<Landscape> (&landscape), _params);
}

boost::optional<Napvig::Trajectory> NapvigRandomized::trajectoryAlgorithm(const Napvig::State &initialState) {
	return followPolicy (initialState, randomizePolicy);
}

RandomizePolicy::RandomizePolicy (const std::shared_ptr<Landscape> _landscape,
								  const std::shared_ptr<NapvigRandomized::Params> &_params):
	Policy(_landscape, _params)
{
	flags.addFlag ("first", false, true);
}

void RandomizePolicy::init () {
	flags.set ("first");
}

Tensor RandomizePolicy::randomize (const torch::Tensor &search)
{
	double thetaRandom = torch::normal (0.0, params().randomizeVariance, {1}).item ().toDouble ();
	Rotation randomizedRotation = Rotation::fromAxisAngle (Rotation::axis2d (), tensor ({thetaRandom}, kDouble));

	return randomizedRotation * search;
}

Tensor RandomizePolicy::getFirstSearch (const Napvig::State &initialState)
{
	if (flags["first"]) {
		flags.reset ("first");

		lastSearch = initialState.search;
		return lastSearch;
	}  else {
		lastSearch = randomize (lastSearch);

		return lastSearch;
	}
}

bool RandomizePolicy::selectTrajectory (const Napvig::Trajectory &trajectory, Termination termination) {
	return (termination != PREDICTION_TERMINATION_COLLISION);
}
