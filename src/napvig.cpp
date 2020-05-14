#include "napvig.h"

#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <torch/autograd.h>
#include <ATen/Layout.h>

using namespace torch;
using namespace torch::indexing;
using namespace std;

Tensor Napvig::nextSample ()
{
	Tensor xStep, ret;
	double taken;
	int num;
	PROFILE_N (taken, [&] {
		// Perform step ahead
		xStep = stepAhead (state.position, state.bearing);

		// Get valley
		ret = valleySearch (xStep, state.bearing, num);
	}, num);
	return ret;
}

Tensor Napvig::projectOnto (const Tensor &space, const Tensor &vector) const {
	return space.mm (space.t ()).mm (vector.unsqueeze (1)).squeeze ();
}

Tensor Napvig::valleySearch (const Tensor &xStep, const Tensor &rSearch, int &num)
{
	// Get basis orthonormal to direction
	Tensor searchSpace = getBaseOrthogonal (rSearch);

	// Initial conditions
	bool terminationCondition = false;
	Tensor xCurr = xStep;
	Tensor gradProject;
	int iterCount;

	//cout << "Start" << endl;
	for (iterCount = 0; !terminationCondition; iterCount++) {
		Tensor gradCurr = map.grad (xCurr);

		// Project gradient on current position
		gradProject = projectOnto (searchSpace, gradCurr);

		// Update rule
		Tensor xNext = xCurr + params.gradientStepSize * gradProject;

		Tensor debugVal =  torch::stack ({map.value (xCurr),gradProject.norm ()}).unsqueeze (0);

		if (iterCount == 0)
			state.gradLog = debugVal;
		else
			state.gradLog = torch::cat ({state.gradLog,debugVal},0);

		double updateDistance = (xNext - xCurr).norm ().item ().toDouble ();

		xCurr = xNext;

		terminationCondition = (updateDistance < params.terminationDistance) || (iterCount > params.terminationCount);
	}

	//cout << "Term after n iter " << iterCount << " w/ norm " << gradProject.norm ()<< endl;
	//cout << xCurr << endl;
	num = iterCount;
	//cout << xCurr << endl;
	return xCurr;
}

Tensor Napvig::stepAhead (const Tensor &xStart, const Tensor &rSearch) const {
	return xStart + params.stepAheadSize * rSearch;
}

Tensor Napvig::getBaseOrthogonal (const Tensor &base) const
{
	Tensor vMatrix;
	std::tie (std::ignore, std::ignore, vMatrix) = base.unsqueeze (0).svd (false);

	return vMatrix.index ({Ellipsis, Slice (1, None)});
}

Tensor Napvig::computeBearing (const Tensor &nextPosition) const {
	return (nextPosition - state.position) / (nextPosition - state.position).norm ();
}

Napvig::Napvig (const NapvigMapParams &mapParams, const NapvigParams &napvigParams):
	map(mapParams),
	params(napvigParams)
{
	resetState ();
	flags.addFlag ("new_measures");
}

void Napvig::step ()
{
	torch::Tensor next = nextSample ();
	state.bearing = getBearing ();
	state.position = next;
	flags.setProcessed ();
}

void Napvig::setMeasures (const Tensor &measures)
{
	resetState ();
	map.setMeasures (measures);
	flags.set("new_measures");
}

double Napvig::mapValue (const Tensor &x) const {
	return map.value (x).item ().toDouble ();
}

Tensor Napvig::getBearing () const {
	return state.bearing;
}

void Napvig::resetState () {
	state.position = torch::tensor ({0, 0}, torch::kDouble);
	state.bearing = torch::tensor ({1, 0}, torch::kDouble);
	state.gradLog = torch::empty (0, torch::kDouble);
}

bool Napvig::isMapReady() const {
	return map.isReady ();
}

bool Napvig::isReady () const {
	return flags.isReady ();
}
