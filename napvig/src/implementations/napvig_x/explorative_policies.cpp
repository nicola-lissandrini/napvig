#include "napvig_x.h"

using namespace std;
using namespace torch;

void ExplorativeCost::convertLandmarks () {
	landmarks->convertAll (framesTracker->current ());
}

ExplorativeCost::ExplorativeCost (const NapvigPredictive::Params::Ptr &_params,
								  const LandmarksBatch::Ptr &_landmarks,
								  const Napvig::FramesTracker::Ptr &_framesTracker):
	CostFunction(_params),
	landmarks(_landmarks),
	framesTracker(_framesTracker)
{
}

double ExplorativeCost::get (const Napvig::Trajectory &trajectory)
{
	double totalCost = 0;

	// Convert landmarks to current frame
	convertLandmarks ();

	for (const Napvig::State &sample : trajectory)
		totalCost += pointCost (sample.position);

	return totalCost;
}

double FullyExplorativeCost::landmarkCostAt (const Tensor &x, const Tensor &landmarkPosition)
{
	const int dims = x.squeeze ().size(0);
	const Tensor xMu = (x.squeeze () - landmarkPosition);
	const double x1 = xMu[0].item().toDouble ();
	const double x2 = xMu[1].item().toDouble ();

	double value = sqrt (pow(2 * M_PI * params().landmarks.radius, (double)dims)) *
			exp (-0.5/params().landmarks.radius * (x1*x1+x2*x2));
	return value;
}

double FullyExplorativeCost::pointCost (const Tensor &point, bool world)
{
	Frame currentFrame;
	double totalCost = 0;

	currentFrame = framesTracker->current ();
	for (const Landmark &landmark : *landmarks) {
		const double exponentialForgetTerm = exp (- params().landmarks.forgettingFactor *
												  landmark.elapsed ());
		const double initialCost = landmarkCostAt (point, world?
													   landmark.getWorldPosition ():
													   landmark.getConvertedPosition ());
		totalCost += exponentialForgetTerm * initialCost;
	}

	return totalCost;
}

FullyExplorativeCost::FullyExplorativeCost (const NapvigPredictive::Params::Ptr &_params,
											const LandmarksBatch::Ptr &_landmarks,
											const Napvig::FramesTracker::Ptr &_framesTracker):
	ExplorativeCost(_params, _landmarks, _framesTracker)
{
}

CostFunction::CostFunction(const NapvigPredictive::Params::Ptr &_paramsData):
	paramsData(_paramsData)
{
}


CostOptimizationPolicy::CostOptimizationPolicy (const Landscape::Ptr &_landscape,
												const NapvigX::Params::Ptr &_params,
												const CostFunction::Ptr &_cost):
	Policy(_landscape, _params),
	cost(_cost)
{
	optimizer = make_shared<GridAngleOptimizer> (cost, params().angleSearch);
}

void CostOptimizationPolicy::setCost(const CostFunction::Ptr &_cost) {
	cost = _cost;
}


void CostOptimizationPolicy::init () {
	optimizer->init ();
}

pair<Tensor, boost::optional<Tensor>> CostOptimizationPolicy::getFirstSearch (const Napvig::State &initialState)
{
	optimizer->setInitialSearch (initialState.search);

	return {optimizer->next (), boost::none};
}

bool CostOptimizationPolicy::processTrajectory (const Napvig::Trajectory &trajectory, Termination termination)
{
	if (termination == PREDICTION_TERMINATION_TARGET_REACHED) {
		finalTrajectoryIndexed = {boost::none, 0};
		return false;
	}

	optimizer->add (make_shared<Napvig::Trajectory> (trajectory), termination == PREDICTION_TERMINATION_COLLISION);

	if (optimizer->isLast ()) {
		auto finalTrajectoryIndexedPtr = optimizer->getOptimal ();

		finalTrajectoryIndexed = {*finalTrajectoryIndexedPtr.first, finalTrajectoryIndexedPtr.second};

		return false;
	}

	return true;
}

IterativeSampledOptimizer::IterativeSampledOptimizer(const CostFunction::Ptr &_cost, int samples):
	cost(_cost)
{
	trajectoriesCosts = torch::empty ({samples});
	trajectories.resize (samples);
}

void IterativeSampledOptimizer::init() {
	current = 0;
	reset ();
}

void IterativeSampledOptimizer::add(const Napvig::Trajectory::Ptr &trajectory, bool infiniteCost)
{
	trajectories[current] = trajectory;
	trajectoriesCosts[current] = infiniteCost ? INFINITY : cost->get (*trajectory);
	current++;
}

std::pair<Napvig::Trajectory::Ptr, int> IterativeSampledOptimizer::getOptimal()
{
	const int argmin = trajectoriesCosts.argmin ().item ().toInt ();

	return {trajectories[argmin], argmin};
}

Tensor SampledAngleOptimizer::getDirection (double angle) {
	Rotation rot = Rotation::fromAxisAngle (angle);
	return rot * initialSearch;
}

void SampledAngleOptimizer::setInitialSearch(const Tensor &_initialSearch) {
	initialSearch = _initialSearch;
}

void GridAngleOptimizer::reset () {
	angle = searchRange.min;
}

Tensor GridAngleOptimizer::next () {
	Tensor nextSearch = getDirection (angle);

	angle += searchRange.step;

	return nextSearch;
}

bool GridAngleOptimizer::isLast() {
	return current == searchRange.count ();
}

double PartiallyExplorativeCost::pointCost(const Tensor &point, bool world)
{
	const double explorativeCost = FullyExplorativeCost::pointCost (point, world);
	const double partiallyExplorativeCost = explorativeCost + params().targetCostWeight * (point - (world? framesTracker->current () : Frame()) * targetFrame->position
																						   ).norm ().item().toDouble ();
	QUA;
	return partiallyExplorativeCost;
}

PartiallyExplorativeCost::PartiallyExplorativeCost (const NapvigPredictive::Params::Ptr &_params,
													const LandmarksBatch::Ptr &_landmarks,
													const Napvig::FramesTracker::Ptr &_framesTracker,
													const Frame::Ptr &_targetFrame):
	FullyExplorativeCost(_params,
						 _landmarks,
						 _framesTracker),
	targetFrame(_targetFrame)
{
	QUA;
}
