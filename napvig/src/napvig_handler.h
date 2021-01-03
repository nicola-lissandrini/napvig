#ifndef NAPVIG_HANDLER_H
#define NAPVIG_HANDLER_H

#include "napvig.h"
#include "implementations/napvig_legacy.h"
#include "implementations/napvig_randomized.h"
#include "implementations/napvig_x/napvig_x.h"
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
		bool stopOnFail;
	};

	std::shared_ptr<Params> paramsData;

	ReadyFlags<std::string> flags;
	std::shared_ptr<Napvig> napvig;
	std::shared_ptr<NapvigDebug> debug;
	CommandPublisher commandPublisherCallback;

	class GetNapvigParams {
		XmlRpc::XmlRpcValue &xmlParams;
		std::shared_ptr<Napvig::Params> paramsData;

		void addCore ();
		void addPredictive ();
		void addRandomized ();
		void addX ();

		template<class NapvigType = Napvig>
		std::shared_ptr<typename NapvigType::Params> params () {
			return std::dynamic_pointer_cast<typename NapvigType::Params> (paramsData);
		}
	public:
		GetNapvigParams (XmlRpc::XmlRpcValue &_xmlParams);

		std::shared_ptr<Napvig::Params> legacy ();
		std::shared_ptr<NapvigRandomized::Params> randomized ();
		std::shared_ptr<NapvigX::Params> x ();
	};
	std::shared_ptr<Params> getNapvigHandlerParams (XmlRpc::XmlRpcValue &xmlParams);
	std::shared_ptr<Landscape::Params> getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams);


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
	void updateTarget (const Frame &targetFrame);
	Napvig::AlgorithmType getType () const;
	std::shared_ptr<NapvigDebug> getDebug();
};

#endif // NAPVIG_HANDLER_H
