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

LidarMeasures::LidarMeasures (const vector<float> &_ranges,
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

shared_ptr<Landscape::Params> NapvigHandler::getLandscapeParams(XmlRpcValue &xmlParams)
{
	shared_ptr<Landscape::Params> landscapeParams = make_shared<Landscape::Params> ();

	landscapeParams->measureRadius = paramDouble (xmlParams, "measure_radius");
	landscapeParams->smoothRadius = paramDouble (xmlParams, "smooth_radius");
	landscapeParams->precision = paramInt (xmlParams,"precision");
	landscapeParams->dim = paramInt(xmlParams,"dimensions");
	landscapeParams->minDistance = paramDouble (xmlParams,"min_distance");

	return landscapeParams;
}

shared_ptr<NapvigHandler::Params> NapvigHandler::getNapvigHandlerParams (XmlRpcValue &xmlParams)
{
	shared_ptr<Params> handlerParams = make_shared<Params> ();

	handlerParams->synchronous = paramBool (xmlParams,"synchronous");

	return handlerParams;
}

void NapvigHandler::GetNapvigParams::addCore()
{
	params->stepAheadSize = paramDouble (xmlParams["core"], "step_ahead_size");
	params->gradientStepSize = paramDouble (xmlParams["core"], "gradient_step_size");
	params->terminationDistance = paramDouble (xmlParams["core"], "termination_distance");
	params->terminationCount = paramInt (xmlParams["core"], "termination_count");
}

void NapvigHandler::GetNapvigParams::addPredictive()
{
	shared_ptr<NapvigPredictive::Params> predictiveParams = dynamic_pointer_cast<NapvigPredictive::Params> (params);

	predictiveParams->windowLength = paramDouble (xmlParams["predictive"], "window_length");
}

void NapvigHandler::GetNapvigParams::addRandomized()
{
	shared_ptr<NapvigRandomized::Params> randomizedParams = dynamic_pointer_cast<NapvigRandomized::Params> (params);

	randomizedParams->randomizeVariance = paramDouble (xmlParams["predictive"]["randomized"], "randomize_variance");
}

NapvigHandler::GetNapvigParams::GetNapvigParams (XmlRpcValue &_xmlParams):
	xmlParams(_xmlParams)
{
}

shared_ptr<Napvig::Params> NapvigHandler::GetNapvigParams::legacy ()
{
	params = make_shared<Napvig::Params> ();

	addCore ();

	return params;
}

shared_ptr<NapvigRandomized::Params> NapvigHandler::GetNapvigParams::randomized ()
{
	params = make_shared<NapvigRandomized::Params> ();

	addCore ();
	addPredictive ();
	addRandomized ();

	return dynamic_pointer_cast<NapvigRandomized::Params> (params);
}

void NapvigHandler::init (Napvig::AlgorithmType type, XmlRpcValue &xmlParams)
{
	shared_ptr<Landscape::Params> landscapeParams = getLandscapeParams (xmlParams["landscape"]);
	GetNapvigParams getParams(xmlParams["napvig"]);
	paramsData = getNapvigHandlerParams (xmlParams);

	switch (type) {
	case Napvig::NAPVIG_LEGACY: {
		shared_ptr<Napvig::Params> napvigParams = getParams.legacy ();
		napvig = make_shared<NapvigLegacy> (landscapeParams,
											napvigParams);
		break;
	}
	case Napvig::NAPVIG_RANDOMIZED: {
		const shared_ptr<NapvigRandomized::Params> napvigRandomizedParams = getParams.randomized ();

		napvig = make_shared<NapvigRandomized> (landscapeParams,
												napvigRandomizedParams);
		break;
	}
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

	if (!params().synchronous)
		dispatchCommand ();
}

Napvig::AlgorithmType NapvigHandler::getType() const {
  return napvig->getType ();
}

int NapvigHandler::synchronousActions()
{
	if (params().synchronous)
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

































