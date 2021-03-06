#include "napvig.h"

#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

using namespace torch;
using namespace torch::indexing;
using namespace std;

/*********************
 * SINGLE STEP IMPLEMENTATION
 * Compute one sample of the trajectory, given the previous
 * Main method: State compute (State q)
 * *******************/

Napvig::Core::Core(Params _params, Landscape &parentLandscape):
	params(_params),
	landscape(parentLandscape)
{}

Tensor Napvig::Core::valleySearch (const Tensor &xStep, const Tensor &rSearch) const
{
	// Get basis orthonormal to direction
	Tensor searchSpace = getBaseOrthogonal (rSearch);

	// Initial conditions
	bool terminationCondition = false;
	Tensor xCurr = xStep;
	Tensor gradProject;
	int iterCount;

	for (iterCount = 0; !terminationCondition; iterCount++) {
		Tensor gradCurr = landscape.grad (xCurr);

		// Project gradient on current position
		gradProject = projectOnto (searchSpace, gradCurr);

		// Update rule
		Tensor xNext = xCurr + params.gradientStepSize * gradProject;

		double updateDistance = (xNext - xCurr).norm ().item ().toDouble ();

		xCurr = xNext;

		terminationCondition = (updateDistance < params.terminationDistance) || (iterCount > params.terminationCount);
	}

	cout << "Term after " << iterCount << " trials" << endl;

	return xCurr;
}

Tensor Napvig::Core::nextSearch(const Tensor &current, const Tensor &next) const {
	return (next - current) / (next - current).norm ();
}

Tensor Napvig::Core::projectOnto (const Tensor &space, const Tensor &vector) const {
	return space.mm (space.t ()).mm (vector.unsqueeze (1)).squeeze ();
}

Tensor Napvig::Core::getBaseOrthogonal (const Tensor &base) const
{
	Tensor vMatrix;
	tie (ignore, ignore, vMatrix) = base.unsqueeze (0).svd (false);

	return vMatrix.index ({Ellipsis, Slice (1, None)});
}

Tensor Napvig::Core::stepAhead (const State &q) const {
	return q.position + params.stepAheadSize * q.search;
}

Napvig::State Napvig::Core::compute (const Napvig::State &q) const
{
	Tensor xStep;
	State next;

	// Perform step ahead
	xStep = stepAhead (q);

	// Get valley
	next.position = valleySearch (xStep, q.search);

	// Get next bearing
	next.search = nextSearch (q.position, next.position);

	return next;
}


/*********************
 * ODOMETRY FRAME UPDATER
 * Update the current frame with respect to the last measure frame
 * On new measures: resetFrame ()
 * On new odom: updateFrame ()
 * *******************/

Napvig::FramesTracker::FramesTracker()
{
	flags.addFlag ("first_frame", true);
	flags.addFlag ("first_reset", true);
}

void Napvig::FramesTracker::resetFrame ()
{
	flags.set ("first_reset");
	measuresFrame = odomFrame;
}

void Napvig::FramesTracker::updateFrame (const Frame &newFrame)
{
	flags.set ("first_frame");

	odomFrame = newFrame;
}

bool Napvig::FramesTracker::isReady () const {
	return flags.isReady ();
}

Napvig::State Napvig::FramesTracker::toMeasuresFrame (const State &stateOdom) {
	Frame transformation = odomFrame.inv () * measuresFrame;

	return State{transformation * stateOdom.position,
				 transformation.orientation * stateOdom.search};
}


/*********************
 * Generic Napvig methods for init and i/o
 * *******************/

Napvig::Napvig (Napvig::AlgorithmType _type,
				const Landscape::Params &landscapeParams,
				const Napvig::Params &napvigParams):
	type(_type),
	landscape(landscapeParams),
	params(napvigParams),
	core(napvigParams, landscape)
{
	debug = make_shared<NapvigDebug> (shared_ptr<Landscape> (&landscape));
}

boost::optional<Napvig::Trajectory> Napvig::computeTrajectory ()
{
	State initialInMeasures = framesTracker.toMeasuresFrame (zeroState);

	return trajectoryAlgorithm (initialInMeasures);
}

void Napvig::setMeasures (const Tensor &measures)
{
	landscape.setMeasures (measures);

	framesTracker.resetFrame ();
}

void Napvig::updateFrame (const Frame &newFrame) {
	framesTracker.updateFrame (newFrame);
}

std::shared_ptr<NapvigDebug> Napvig::getDebug() {
	return debug;
}

bool Napvig::isReady () const {
	return landscape.isReady () && framesTracker.isReady ();
}

Napvig::AlgorithmType Napvig::getType() const {
	return type;
}

/********
 * Debug info
 * ******/
NapvigDebug::NapvigDebug(const std::shared_ptr<Landscape> &_landscape):
	landscape(_landscape)
{
}


































