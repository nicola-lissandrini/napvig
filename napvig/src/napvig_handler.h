#ifndef NAPVIG_HANDLER_H
#define NAPVIG_HANDLER_H

#include "napvig.h"
#include "implementations/napvig_legacy.h"
#include "implementations/napvig_randomized.h"
#include <xmlrpcpp/XmlRpc.h>

class Measures
{
public:
	virtual torch::Tensor get () = 0;
};

class LidarMeasures : public Measures
{
	std::vector<float> ranges;
	double angleMin, angleIncrement;
	torch::Tensor polar2rectangularMeasure (double radius, double angle);
public:
	LidarMeasures (const std::vector<float> &_ranges,
				   double _angleMin, double _angleIncrement);
	torch::Tensor get ();
};

typedef std::function<void (const torch::Tensor &)> CommandPublisher;

class NapvigHandler
{
	struct Params {
		bool synchronous;
	};

	std::shared_ptr<Params> paramsData;

	ReadyFlags<std::string> flags;
	std::shared_ptr<Napvig> napvig;
	std::shared_ptr<NapvigDebug> debug;
	CommandPublisher commandPublisherCallback;

	std::shared_ptr<Landscape::Params> getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams);

	class GetNapvigParams {
		XmlRpc::XmlRpcValue &xmlParams;
		std::shared_ptr<Napvig::Params> params;

		void addCore ();
		void addPredictive ();
		void addRandomized ();

	public:
		GetNapvigParams (XmlRpc::XmlRpcValue &_xmlParams);

		std::shared_ptr<Napvig::Params> legacy ();
		std::shared_ptr<NapvigRandomized::Params> randomized ();
	};

	std::shared_ptr<Napvig::Params> addNapvigParams (const std::shared_ptr<Napvig::Params> &napvigParams, XmlRpc::XmlRpcValue &xmlParams);
	std::shared_ptr<NapvigPredictive::Params> addNapvigPredictiveParams (const std::shared_ptr<Napvig::Params> &napvigParams, XmlRpc::XmlRpcValue &xmlParams);

	std::shared_ptr<Params> getNapvigHandlerParams (XmlRpc::XmlRpcValue &xmlParams);
	std::shared_ptr<NapvigRandomized::Params> getNapvigRandomizedParams (XmlRpc::XmlRpcValue &xmlParams);

	void dispatchCommand ();
	boost::optional<torch::Tensor> getCommand();

	const Params &params() {
		return *paramsData;
	}

public:
	NapvigHandler (CommandPublisher _commandPublisherCallback);

	void init (Napvig::AlgorithmType type,
			   XmlRpc::XmlRpcValue &xmlParams);

	int synchronousActions ();
	void updateMeasures (std::shared_ptr<Measures> measures);
	void updateFrame (const Frame &odomFrame);
	Napvig::AlgorithmType getType () const;
	std::shared_ptr<NapvigDebug> getDebug();
};

#endif // NAPVIG_HANDLER_H
