#ifndef NAPVIG_HANDLER_H
#define NAPVIG_HANDLER_H

#include "napvig.h"
#include "implementations/napvig_legacy.h"
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
	} handlerParams;

	ReadyFlags<std::string> flags;
	std::shared_ptr<Napvig> napvig;
	std::shared_ptr<NapvigDebug> debug;
	CommandPublisher commandPublisherCallback;

	Landscape::Params getLandscapeParams (XmlRpc::XmlRpcValue &handlerParams);
	Napvig::Params getNapvigParams (XmlRpc::XmlRpcValue &handlerParams);
	Params getNapvigHandlerParams (XmlRpc::XmlRpcValue &handlerParams);

	void dispatchCommand ();
	boost::optional<torch::Tensor> getCommand();

public:
	NapvigHandler (CommandPublisher _commandPublisherCallback);

	void init (Napvig::AlgorithmType type,
			   XmlRpc::XmlRpcValue &handlerParams);

	int synchronousActions ();
	void updateMeasures (std::shared_ptr<Measures> measures);
	void updateFrame (const Frame &odomFrame);
	Napvig::AlgorithmType getType () const;
	std::shared_ptr<NapvigDebug> getDebug();
};

#endif // NAPVIG_HANDLER_H
