#ifndef NAPVIG_PREDICTIVE_H
#define NAPVIG_PREDICTIVE_H

#include "../napvig.h"
#include "policy.h"


class NapvigPredictive : public Napvig
{
public:
	struct Params {
		Napvig::Params napvigParams;
		int lookaheadHorizon;
	};

private:
	Params params;
	ReadyFlags<std::string> flags;
	std::shared_ptr<Policy> policy;

protected:
	void setPolicy (std::shared_ptr<Policy> _policy);

	// This class is not meant to be instantiated directy
	// A subclass is needed to specify the policy
	NapvigPredictive (AlgorithmType type, const Landscape::Params &landscapeParams,
			 const NapvigPredictive::Params &predictiveParams);

	std::pair<Napvig::Trajectory, bool> predictTrajectory (const Napvig::State &initialState, const std::shared_ptr<Policy> &policy);
	boost::optional<Napvig::Trajectory> trajectoryAlgorithm (const State &initialState);
};

#endif // NAPVIG_PREDICTIVE_H
