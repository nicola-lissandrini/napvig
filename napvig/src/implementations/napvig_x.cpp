#include "napvig_x.h"

using namespace std;
using namespace torch;

NapvigX::NapvigX (const std::shared_ptr<Landscape::Params> &landscapeParams,
				  const std::shared_ptr<NapvigX::Params> &params):
	NapvigPredictive(NAPVIG_X,
					 landscapeParams,
					 params)
{
	fextPolicy = make_shared<FullyExploitative> ();
	fexpPolicy = make_shared<FullyExplorative> ();
	pextPolicy = make_shared<PartiallyExploitative> ();
}

boost::optional<Napvig::Trajectory> NapvigX::trajectoryAlgorithm (const State &initialState)
{
	if (!this->isReady ())
		return boost::none;

	boost::optional<Napvig::Trajectory> trajectory;

	// Check an existing Fully Exploitative trajectory exists
	trajectory = followPolicy (initialState, fextPolicy);
	if (trajectory)
		return trajectory;

	// Otherwise
	// boh

}



