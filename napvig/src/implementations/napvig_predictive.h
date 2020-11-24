#ifndef NAPVIG_PREDICTIVE_H
#define NAPVIG_PREDICTIVE_H

#include "../napvig.h"
#include "policy.h"


class NapvigPredictive : public Napvig
{
private:
	Params params;
	ReadyFlags<std::string> flags;

	std::pair<Napvig::Trajectory, bool> predictTrajectory (const Napvig::State &initialState, const std::shared_ptr<Policy> &policy);
	boost::optional<Napvig::Trajectory> policyTrajectory (const State &initialState, const std::shared_ptr<Policy> &policy);

public:
	NapvigPredictive (AlgorithmType type,
					  const std::shared_ptr<Landscape::Params> &landscapeParams,
					  const std::shared_ptr<Napvig::Params> &predictiveParams);
};

#endif // NAPVIG_PREDICTIVE_H
