#include "napvig.h"

#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <torch/autograd.h>
#include <ATen/Layout.h>

using namespace torch;
using namespace torch::indexing;
using namespace std;

Napvig::Napvig (const NapvigMapParams &mapParams, const NapvigParams &napvigParams):
	map(mapParams),
	params(napvigParams),
	state{initialPosition, initialSearch},
	frame(1,0),
	oldFrame(1,0)
{
	flags.addFlag ("new_measures");
	flags.addFlag ("frame_updated");
}

pair<Napvig::State, Tensor> Napvig::nextSample (const State &q) const
{
	Tensor xStep;
	State next;

	int num;

	// Perform step ahead
	xStep = stepAhead (q);

	// Get valley
	Tensor gradLog;
	tie (next.position, gradLog) = valleySearch (xStep, q.search, num);

	// Get next bearing
	next.search = computeBearing (q.position, next.position);

	return {next, gradLog};
}

Tensor Napvig::projectOnto (const Tensor &space, const Tensor &vector) const {
	return space.mm (space.t ()).mm (vector.unsqueeze (1)).squeeze ();
}
pair<Tensor, Tensor> Napvig::valleySearch(const Tensor &xStep, const Tensor &rSearch, int &num) const
{
	// Get basis orthonormal to direction
	Tensor searchSpace = getBaseOrthogonal (rSearch);

	// Initial conditions
	bool terminationCondition = false;
	Tensor xCurr = xStep;
	Tensor gradProject;
	Tensor gradLog;
	int iterCount;

	//cout << "Start" << endl;
	for (iterCount = 0; !terminationCondition; iterCount++) {
		Tensor gradCurr = map.grad (xCurr);

		// Project gradient on current position
		gradProject = projectOnto (searchSpace, gradCurr);

		// Update rule
		Tensor xNext = xCurr + params.gradientStepSize * gradProject;

		Tensor debugVal =  torch::stack ({map.value (xCurr),gradProject.norm ()}).unsqueeze (0);

		// For debug
		if (iterCount == 0)
			gradLog = debugVal;
		else
			gradLog = torch::cat ({gradLog,debugVal},0);
		double updateDistance = (xNext - xCurr).norm ().item ().toDouble ();

		xCurr = xNext;

		terminationCondition = (updateDistance < params.terminationDistance) || (iterCount > params.terminationCount);
	}

	num = iterCount;
	return {xCurr, gradLog};
}

Tensor Napvig::stepAhead (const State &q) const {
	return q.position + params.stepAheadSize * q.search;
}

Tensor Napvig::getBaseOrthogonal (const Tensor &base) const
{
	Tensor vMatrix;
	tie (ignore, ignore, vMatrix) = base.unsqueeze (0).svd (false);

	return vMatrix.index ({Ellipsis, Slice (1, None)});
}

Tensor Napvig::computeBearing (const Tensor &oldPosition, const Tensor &nextPosition) const {
	return (nextPosition - oldPosition) / (nextPosition - oldPosition).norm ();
}

bool Napvig::collides (const Tensor &x) const {
	return mapValue (x) > map.gammaDistance (params.minDistance);
}


Napvig::State Napvig::stepSingle()
{
	Tensor step, gradLog;
	tie (step, gradLog) = nextSample (state);
	return step;
}

complex<double> vec2complex (const Tensor &tensor) {
	return tensor[0].item().toDouble () + 1i * tensor[1].item().toDouble ();
}

Tensor complex2vec (complex<double> val) {
	Tensor ret = torch::empty ({2}, kDouble);
	ret[0] = real(val);
	ret[1] = imag(val);
	return ret;
}

// Rotate the search direction by a gaussian distributed random angle of variance `params.scatterVariance`
Napvig::State Napvig::randomize (const State &state) const
{
	State randomized;
	double thetaRandom = torch::normal (0.0, params.scatterVariance, {1}).item().toDouble ();
	complex<double> searchComplex = vec2complex (state.search);

	randomized.search = complex2vec (polar(1.0,thetaRandom) * searchComplex);
	randomized.position = state.position;

	return randomized;
}

tuple<Napvig::State, bool, Tensor> Napvig::predictCollision (const State &initialState, int maxCount) const
{
	Napvig::State first, curr;
	Tensor path;
	int stepCount = 0;
	bool collision;

	// Compute first step...
	tie (first, ignore) = nextSample (initialState);

	curr = first;
	collision = collides (first.position);
	path = first.position.unsqueeze (0);

	// ...and check if it would lead to collision in 'maxCount' steps
	while (!collision && stepCount < maxCount) {
		tie (curr, ignore) = nextSample (curr);
		collision = collides (curr.position);
		path = torch::cat ({path, curr.position.unsqueeze (0)},1);

		if (collision)
			cout << "Collision in prediction at sample " << stepCount << endl;

		stepCount++;
	}

	// if not, return first step
	return {first, collision, path};
}

pair<Napvig::State, SearchHistory> Napvig::stepDiscovery()
{
	// Perform first deterministic step
	State step;
	SearchHistory searchHistory;
	Tensor currPath;
	bool collision;

	tie (step, collision) = predictCollision (state, params.lookaheadHorizon);

	// If no collision take the first step of the predicted path
	if (!collision) {
		return step;
	}

	State randomized = state;
	int trialNeeded = 1;

	// Otherwise take a random direction and predict again
	while (collision) {
		randomized = randomize (randomized);
		cout << "Trying another direction" << endl;
		tie (step, collision) = predictCollision (randomized, params.lookaheadHorizon);
		trialNeeded++;
	}

	cout << "Found a way. Took " << trialNeeded << " random trials.";

	// Take the last step
	return step;
}

void Napvig::step ()
{
	switch (params.algorithm) {
	case NAPVIG_SINGLE_STEP:
		state = stepSingle ();
		break;
	case NAPVIG_PREDICT_COLLISION:
		state = stepDiscovery ();
		break;
	}
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

Tensor Napvig::mapGrad(const Tensor &x) const {
	return map.grad (x);
}

Tensor Napvig::getPosition() const {
	return state.position;
}

Tensor Napvig::getBearing() const {
	return state.search;
}

void Napvig::resetState ()
{
	oldState = state;
	state.position = initialPosition;

	// Keep last search direction in the new frame got from odometry
	//cout << "Old " << state.search << endl;
	cout << "Diff "  << imag (log (oldFrame / frame)) << endl;
	state.search = complex2vec (oldFrame / frame * vec2complex (state.search));
	//cout << "New " << state.search << endl;

	gradLog = torch::empty (0, torch::kDouble);
}

void Napvig::updateFrame (complexd newFrame)
{
	oldFrame = frame;
	frame = newFrame;

	flags.set ("frame_updated");
}

bool Napvig::isMapReady() const {
	return map.isReady ();
}

bool Napvig::isReady () const {
	return map.isReady () && flags.isReady ();
}
