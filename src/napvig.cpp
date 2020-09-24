#include "napvig.h"

#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

using namespace torch;
using namespace torch::indexing;
using namespace std;

Napvig::Napvig (const NapvigMap::Params &mapParams, const Napvig::Params &napvigParams):
	map(mapParams),
	params(napvigParams),
	state{initialPosition, initialSearch}
{

	flags.addFlag ("first_measure", true);
	flags.addFlag ("first_frame", true);
	flags.addFlag ("new_measures");
	flags.addFlag ("frame_updated");
}

Napvig::State Napvig::nextSample (const State &q) const
{
	Tensor xStep;
	State next;

	int num;

	// Perform step ahead
	xStep = stepAhead (q);

	// Get valley
	next.position = valleySearch (xStep, q.search, num);

	// Get next bearing
	next.search = computeBearing (q.position, next.position);

	return next;
}

Tensor Napvig::projectOnto (const Tensor &space, const Tensor &vector) const {
	return space.mm (space.t ()).mm (vector.unsqueeze (1)).squeeze ();
}

// Core algorithm
Tensor Napvig::valleySearch (const Tensor &xStep, const Tensor &rSearch, int &num) const
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

		double updateDistance = (xNext - xCurr).norm ().item ().toDouble ();

		xCurr = xNext;

		terminationCondition = (updateDistance < params.terminationDistance) || (iterCount > params.terminationCount);
	}

	num = iterCount;
	return xCurr;
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
	return nextSample (state);
}

Napvig::State Napvig::randomize (const State &state) const
{
	State randomized;
	double thetaRandom = torch::normal (0.0, params.scatterVariance, {1}).item().toDouble ();
	// Only works with 2D
	Rotation randomizedRotation = Rotation::fromAxisAngle (Rotation::axis2d (), tensor ({thetaRandom},kDouble));

	randomized.search = randomizedRotation * state.search;
	randomized.position = state.position;

	return randomized;
}

tuple<Napvig::State, bool, Tensor> Napvig::predictCollision (const State &initialState, int maxCount) const
{
	Napvig::State first, curr;
	Tensor history;
	int stepCount = 0;
	bool collision;

	// Compute first step...
	first = nextSample (initialState);

	curr = first;
	collision = collides (first.position);
	history = first.position.unsqueeze (0);

	// ...and check if it would lead to collision in 'maxCount' steps
	while (!collision && stepCount < maxCount) {
		curr = nextSample (curr);
		collision = collides (curr.position);
		history = torch::cat ({history, curr.position.unsqueeze (0)},0);

		if (collision)
			cout << "Collision in prediction at sample " << stepCount << endl;

		stepCount++;
	}

	// if not, return first step
	return {first, collision, history};
}

pair<Napvig::State, Napvig::SearchHistory> Napvig::stepDiscovery()
{
	// Perform first deterministic step
	State step;
	SearchHistory history;
	Tensor currHistory;

	bool collision;

	tie (step, collision, currHistory) = predictCollision (state, params.lookaheadHorizon);
	history.triedPaths.push_back (currHistory);
	history.initialSearches.push_back (state.search);
	
	// If no collision take the first step of the predicted path
	if (!collision)
		return {step, history};

	State randomized = state;
	int trialNeeded = 1;

	// Otherwise take a random direction and predict again
	while (collision) {
		randomized = randomize (randomized);

		tie (step, collision, currHistory) = predictCollision (randomized, params.lookaheadHorizon);
		history.triedPaths.push_back (currHistory);
		history.initialSearches.push_back (randomized.search);
		trialNeeded++;
	}

	// Take the last step
	return {step, history};
}

void Napvig::keepLastSearch () {
	oldState = state;
	state.search = setpoint.search;
}

bool Napvig::step ()
{
	// Wait until at least one between measures and frame are updated
	if (!flags["new_measures"] && !flags["frame_updated"])
		return false;

	updatePosition ();
	updateOrientation ();


	switch (params.algorithm) {
	case SINGLE_STEP:
		setpoint = stepSingle ();
		break;
	case PREDICT_COLLISION:
		tie (setpoint, lastHistory) = stepDiscovery ();
		break;
	}

	// Update last search direction as state
	keepLastSearch ();

	flags.setProcessed ();

	return true;
}

void Napvig::setMeasures (const Tensor &measures)
{
	map.setMeasures (measures);

	flags.set ("first_measure");
	flags.set ("new_measures");
}

double Napvig::mapValue (const Tensor &x) const {
	return map.value (x).item ().toDouble ();
}

Tensor Napvig::mapGrad(const Tensor &x) const {
	return map.grad (x);
}

Tensor Napvig::getSetpointPosition() const {
	return setpoint.position;
}

Tensor Napvig::getSetpointDirection() const {
	return setpoint.search;
}

Napvig::SearchHistory Napvig::getSearchHistory() const {
	return lastHistory;
}

void Napvig::updateOrientation () {
	oldState = state;
	if (flags["frame_updated"]) {
		state.search = frame.orientation.inv () * oldFrame.orientation * state.search;
	}
}

void Napvig::updatePosition () {
	oldState = state;
	if (flags["new_measures"]) {
		// If measures have been update, reset the map
		state.position = initialPosition.clone ();
	} else {
		// Otherwise, the robot has moved before the new scan, so we go open loop:
		// This is reasonable if the scan rate is sufficiently higher than the motion
		// of the obstacles
		state.position += frame.position - oldFrame.position;
	}
}

void Napvig::updateFrame (Frame newFrame)
{
	oldFrame = frame;
	frame = newFrame;

	flags.set ("first_frame");
	flags.set ("frame_updated");
}

void Napvig::updateTarget (Frame targetFrame)
{
	// Not implemented yet
}

bool Napvig::isMapReady() const {
	return map.isReady ();
}

bool Napvig::isReady () const {
	return map.isReady () && flags["first_measure"] && flags["first_frame"];
}

int Napvig::getDim() const {
	return map.getDim ();
}
