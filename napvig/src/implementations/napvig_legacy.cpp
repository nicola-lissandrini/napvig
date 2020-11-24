#include "napvig_legacy.h"

#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

using namespace torch;
using namespace torch::indexing;
using namespace std;

NapvigLegacy::NapvigLegacy(const Landscape::Params &landscapeParams, const Params &napvigParams):
	Napvig(NAPVIG_LEGACY,
		   landscapeParams,
		   napvigParams)
{}

boost::optional<Napvig::Trajectory> NapvigLegacy::trajectoryAlgorithm (const State &initialState)
{
	boost::optional<Trajectory> trajectory;

	if (!this->isReady ())
		return boost::none;

	State step = core.compute (initialState);
	trajectory = Trajectory {step};

	return trajectory;
}
