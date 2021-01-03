#include "napvig_x.h"

using namespace std;
using namespace torch;

void ExplorativeCost::convertLandmarks () {
	landmarks->convertAll (framesTracker->current ());
}

ExplorativeCost::ExplorativeCost (const shared_ptr<NapvigPredictive::Params> &_params,
								  const shared_ptr<LandmarksBatch> &_landmarks,
								  const shared_ptr<Napvig::FramesTracker> &_framesTracker):
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

double FullyExplorativeCost::pointCost (const Tensor &point)
{
	double taken;
	Frame currentFrame;
	double totalCost = 0;
/*
	int QUANTI = landmarks->size ();
	cout <<	"+++ MA QUANTI " << QUANTI << endl;
*/
	//PROFILE_N (taken,[&]{
	currentFrame = framesTracker->current ();
	for (const Landmark &landmark : *landmarks) {
		const double exponentialForgetTerm = exp (- params().landmarks.forgettingFactor *
												  landmark.elapsed ());
		const double initialCost = landmarkCostAt (point, landmark.getConvertedPosition ());
		totalCost += exponentialForgetTerm * initialCost;
	}
	//}, QUANTI);
	return totalCost;
}

FullyExplorativeCost::FullyExplorativeCost (const std::shared_ptr<NapvigPredictive::Params> &_params,
											const std::shared_ptr<LandmarksBatch> &_landmarks,
											const std::shared_ptr<Napvig::FramesTracker> &_framesTracker):
	ExplorativeCost(_params, _landmarks, _framesTracker)
{
}

CostFunction::CostFunction(const std::shared_ptr<NapvigPredictive::Params> &_paramsData):
	paramsData(_paramsData)
{
}


CostOptimizationPolicy::CostOptimizationPolicy (const shared_ptr<Landscape> &_landscape,
												const shared_ptr<NapvigX::Params> &_params,
												const std::shared_ptr<CostFunction> &_cost):
	Policy(_landscape, _params),
	cost(_cost)
{
	optimizer = make_shared<GridAngleOptimizer> (cost, params().angleSearch);
}

void CostOptimizationPolicy::setCost(const std::shared_ptr<CostFunction> &_cost) {
	cost = _cost;
}


void CostOptimizationPolicy::init () {
	QUA;
	optimizer->init ();
	QUA;
}

pair<Tensor, boost::optional<Tensor>> CostOptimizationPolicy::getFirstSearch (const Napvig::State &initialState)
{
	Tensor nextFirstSearch = optimizer->next ();

	return {nextFirstSearch, boost::none};
}

bool CostOptimizationPolicy::processTrajectory (const Napvig::Trajectory &trajectory, Termination termination)
{
	optimizer->add (make_shared<Napvig::Trajectory> (trajectory));

	return !optimizer->isLast ();
}

IterativeSampledOptimizer::IterativeSampledOptimizer(const std::shared_ptr<CostFunction> &_cost, int samples):
	cost(_cost)
{
	trajectoriesCosts = torch::empty ({samples});
	trajectories.resize (samples);
}

void IterativeSampledOptimizer::init() {
	current = 0;
	reset ();
}

void IterativeSampledOptimizer::add(const std::shared_ptr<Napvig::Trajectory> &trajectory)
{
	trajectories[current] = trajectory;
	trajectoriesCosts[current] = cost->get (*trajectory);
	current++;
}

std::pair<std::shared_ptr<Napvig::Trajectory>, int> IterativeSampledOptimizer::getOpt()
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
