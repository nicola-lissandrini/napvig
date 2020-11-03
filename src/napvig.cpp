#include "napvig.h"

#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

using namespace torch;
using namespace torch::indexing;
using namespace std;

Napvig::Napvig (const NapvigMap::Params &mapParams, const Napvig::Params &napvigParams, NapvigDebug *_debug):
	map(mapParams),
	params(napvigParams),
	state{initialPosition, initialSearch},
	debug(_debug),
	mode(EXPLOITATION),
	targetUnreachable(false)
{
	flags.addFlag ("first_measure", true);
	flags.addFlag ("first_frame", true);
	flags.addFlag ("first_target", true);
	flags.addFlag ("first_corridor", true);
	flags.addFlag ("new_measures");
	flags.addFlag ("frame_updated");
	flags.addFlag ("target_updated");
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

	debug->values.resize (0);

	// Initial conditions
	bool terminationCondition = false;
	Tensor xCurr = xStep;
	Tensor gradProject;
	int iterCount;

	for (iterCount = 0; !terminationCondition; iterCount++) {
		Tensor gradCurr = map.grad (xCurr);

		// Project gradient on current position
		gradProject = projectOnto (searchSpace, gradCurr);

		// Update rule
		Tensor xNext = xCurr + params.gradientStepSize * gradProject;

		double updateDistance = (xNext - xCurr).norm ().item ().toDouble ();

		xCurr = xNext.clone ();

		debug->values.push_back (gradProject.norm ().item().toDouble ());

		terminationCondition = (updateDistance < params.terminationDistance) || (iterCount > params.terminationCount);
	}

	//cout << "Iter count " << iterCount << endl;

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


Napvig::State Napvig::stepSingle (State initialState)
{
	return nextSample (initialState);
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

tuple<Napvig::State, bool, Tensor> Napvig::predictTrajectory (const State &initialState, int maxCount) const
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

		stepCount++;
	}

	// if not, return first step
	return {first, collision, history};
}

pair<Napvig::State, Napvig::SearchHistory> Napvig::stepRandomizedRecovery (State initialState)
{
	// Perform first deterministic step
	State step;
	SearchHistory history;
	Tensor currHistory;

	bool collision;

	tie (step, collision, currHistory) = predictTrajectory (state, params.lookaheadHorizon);
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

		tie (step, collision, currHistory) = predictTrajectory (randomized, params.lookaheadHorizon);
		history.triedPaths.push_back (currHistory);
		history.initialSearches.push_back (randomized.search);
		trialNeeded++;
	}

	// Take the last step
	return {step, history};
}

// Perform N predictions with N initial directions and choose the best one
pair<Napvig::State, Napvig::SearchHistory> Napvig::stepOptimizedTrajectory (State initialState)
{
	State step, current;
	Tensor trajectory;
	vector<State> steps;
	Tensor costs;
	bool collision;
	int count = 0;
	SearchHistory history;

	current = initialState;

	// Initialize empty vector for trajectory costs
	costs = torch::empty ({(int)floor((params.trajectoryOptimizerParams.rangeAngleMax -
										params.trajectoryOptimizerParams.rangeAngleMin) /
										params.trajectoryOptimizerParams.rangeAngleStep)}, kDouble);

	// Grid search on angles:
	for (double currAngle = params.trajectoryOptimizerParams.rangeAngleMin;
		 currAngle < params.trajectoryOptimizerParams.rangeAngleMax;
		 currAngle += params.trajectoryOptimizerParams.rangeAngleStep, count ++)
	{
		Rotation currRotation = Rotation::fromAxisAngle (Rotation::axis2d (), tensor({currAngle}, kDouble));
		current.search = currRotation * initialState.search;
		tie (step, collision, trajectory) = predictTrajectory (current, params.lookaheadHorizon);

		steps.push_back (step);
		if (collision)
			costs[count] = INFINITY;
		else
			costs[count] = evaluateCost (trajectory);

		// Debug purposes - keep track of tried trajectories
		history.triedPaths.push_back (trajectory);
		history.initialSearches.push_back (step.search);


		cout << "Angle " << count << " score " << costs[count].item ().toDouble () << endl;
	}

	// Choose step corresponding to trajectory with maximum score
	int bestIndex = costs.argmin ().item ().toInt ();

	// Save to history (for debug)
	history.chosen = bestIndex;

	cout << "Chosen " << bestIndex << endl; // " " << params.trajectoryOptimizerParams.rangeAngleMin + double (bestIndex) * params.trajectoryOptimizerParams.rangeAngleStep <<  endl;

	return {steps[bestIndex], history};
}

void Napvig::keepLastSearch () {
	oldState = state.clone ();
	state.search = setpoint.search.clone ();
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
		setpoint = stepSingle (state);
		break;
	case RANDOMIZED_RECOVERY:
		tie (setpoint, lastHistory) = stepRandomizedRecovery (state);
		break;
	case OPTIMIZED_TRAJECTORY:
		tie(setpoint, lastHistory) = stepOptimizedTrajectory (state);
		break;
	}


	// Update last search direction as state
	if (params.keepLastSearch)
		keepLastSearch ();

	flags.setProcessed ();

	return true;
}



double Napvig::evaluateCost (const Tensor &trajectory) const
{
	switch (mode) {
	case EXPLORATION:
		// Not implemented yet
		return NAN;
	case EXPLOITATION:
		return costDistanceToTarget (trajectory);
	}

	return NAN; // suppress warning
}


double Napvig::costPathLength (const Tensor &trajectory) const
{
	Tensor sum = tensor ({0},kDouble);

	for (int i = 1; i < trajectory.size(0); i++) {
		sum +=  (trajectory[i] - trajectory[i-1]).norm ();
	}

	return sum.item ().toDouble ();
}

double Napvig::costDistanceToTarget (const Tensor &trajectory) const
{
	Tensor sum = tensor ({0}, kDouble);
	for (int i = 0; i < trajectory.size(0); i++) {
		sum += (targetFrame.position - trajectory[i]).norm ();
	}

	return sum.item().toDouble ();
}

void Napvig::setMeasures (const Tensor &measures)
{
	map.setMeasures (measures);
	state.search = initialSearch.clone ();
	state.position = initialPosition.clone ();

	flags.set ("first_measure");
	flags.set ("new_measures");
}

void Napvig::setCorridor(const Tensor &corridor)
{
	debug->corridor = corridor;

	flags.set ("first_corridor");
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
	oldState = state.clone ();
	if (flags["frame_updated"]) {
		state.search = frame.orientation.inv () * oldFrame.orientation * state.search;
	}
}

void Napvig::updatePosition () {
	oldState = state.clone ();
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
	oldFrame = frame.clone ();
	frame = newFrame.clone ();

	flags.set ("first_frame");
	flags.set ("frame_updated");
}

void Napvig::updateTarget (Frame newTargetFrame)
{
	targetFrame = newTargetFrame;

	flags.set ("first_target");
	flags.set ("target_updated");
}

double Napvig::getDistanceFromCorridor()
{
	if (!flags["first_corridor"])
		return NAN;

	Tensor worldPos = frame.orientation * setpoint.position + frame.position;

	return (worldPos.unsqueeze (0) - debug->corridor).norm (nullopt, 1).min ().item ().toDouble ();
}

bool Napvig::isMapReady() const {
	return map.isReady ();
}

bool Napvig::isReady () const {
	// NOTE: first_target only if full_exploit is used
	return map.isReady () && flags["first_measure"] && flags["first_frame"] && flags["first_target"];
}

bool Napvig::isTargetUnreachable() const {
	return targetUnreachable;
}

int Napvig::getDim() const {
	return map.getDim ();
}
