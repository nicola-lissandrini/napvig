#include "napvig_handler.h"

using namespace std;
using namespace XmlRpc;
using namespace torch;
using namespace torch::indexing;

/*****************
 * Measures conversions definitiosn
 * ***************/

torch::Tensor LidarMeasures::polar2rectangularMeasure (double radius, double angle) {
	return torch::tensor ({radius * cos(angle), radius*sin(angle)});
}

LidarMeasures::LidarMeasures (const std::vector<float> &_ranges,
							  double _angleMin, double _angleIncrement):
	ranges(_ranges),
	angleMin(_angleMin),
	angleIncrement(_angleIncrement)
{
}

Tensor LidarMeasures::get ()
{
	const int measCount = ranges.size ();
	Tensor measures = torch::empty ({measCount, N_DIM});

	for (int i = 0; i < measCount; i++) {
		double currAngle = angleMin + i * angleIncrement;
		double currRadius = ranges[i];

		measures.index_put_ ({i, None}, polar2rectangularMeasure (currRadius, currAngle));
	}

	return measures;
}

/*****************
 * Handler definitions
 * ***************/

NapvigHandler::NapvigHandler(CommandPublisher _commandPublisherCallback):
	commandPublisherCallback(_commandPublisherCallback)
{
	flags.addFlag ("first_params");
}

Landscape::Params NapvigHandler::getLandscapeParams(XmlRpcValue &params)
{
	Landscape::Params lanscapeParams;

	lanscapeParams.measureRadius = paramDouble (params, "measure_radius");
	lanscapeParams.smoothRadius = paramDouble (params, "smooth_radius");
	lanscapeParams.precision = paramInt (params,"precision");
	lanscapeParams.dim = paramInt(params,"dimensions");
	lanscapeParams.minDistance = paramDouble (params,"min_distance");

	return lanscapeParams;
}

Napvig::Params NapvigHandler::getNapvigParams(XmlRpcValue &params)
{
	Napvig::Params napvigParams;

	napvigParams.stepAheadSize = paramDouble (params, "step_ahead_size");
	napvigParams.gradientStepSize = paramDouble (params, "gradient_step_size");
	napvigParams.terminationDistance = paramDouble (params, "termination_distance");
	napvigParams.terminationCount = paramInt (params, "termination_count");

	return napvigParams;
}

NapvigHandler::Params NapvigHandler::getNapvigHandlerParams (XmlRpcValue &params)
{
	Params handlerParams;

	handlerParams.synchronous = paramBool (params,"synchronous");

	return handlerParams;
}

void NapvigHandler::init (Napvig::AlgorithmType type, XmlRpcValue &params)
{
	Landscape::Params landscapeParams = getLandscapeParams (params["landscape"]);
	Napvig::Params napvigParams = getNapvigParams (params["napvig"]["core"]);
	handlerParams = getNapvigHandlerParams (params);

	switch (type) {
	case Napvig::NAPVIG_LEGACY:
		napvig = make_shared<NapvigLegacy> (landscapeParams,
											napvigParams);
		break;
	default:
		assert (false && "Not implemented");
		break;
	}
	flags.set ("first_params");
}

void NapvigHandler::dispatchCommand ()
{
	boost::optional<Tensor> command = getCommand ();

	if (command)
		commandPublisherCallback (command.value ());
}

boost::optional<torch::Tensor> NapvigHandler::getCommand ()
{
	boost::optional<torch::Tensor> command;
	auto trajectory = napvig->computeTrajectory ();

	if (trajectory)
		// Get first trajectory position sample
		command = trajectory->at (0).position;

	return command;
}

void NapvigHandler::updateMeasures (shared_ptr<Measures> measures) {
	napvig->setMeasures (measures->get ());
}

void NapvigHandler::updateFrame (const Frame &odomFrame) {
	napvig->updateFrame (odomFrame);

	if (!handlerParams.synchronous)
		dispatchCommand ();
}

Napvig::AlgorithmType NapvigHandler::getType() const {
  return napvig->getType ();
}

int NapvigHandler::synchronousActions()
{
	if (handlerParams.synchronous)
		dispatchCommand ();

	return 0;
}

shared_ptr<NapvigDebug> NapvigHandler::getDebug()
{
	if (napvig)
		return napvig->getDebug ();
	else
		return shared_ptr<NapvigDebug> ();
}































