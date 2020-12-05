#ifndef NAPVIG_PREDICTIVE_H
#define NAPVIG_PREDICTIVE_H

#include "../napvig.h"
#include "policy.h"


class NapvigPredictive : public Napvig
{
public:
	struct Params : Napvig::Params {
		int windowLength;
	};


protected:
	std::pair<Napvig::Trajectory, PolicyAbstract<NapvigPredictive::Params>::Termination> predictTrajectory (const State &initialState, const std::shared_ptr<PolicyAbstract<NapvigPredictive::Params>> &policy);
	boost::optional<Napvig::Trajectory> followPolicy (const State &initialState, const std::shared_ptr<PolicyAbstract<NapvigPredictive::Params>> &policy);

public:
	NapvigPredictive (AlgorithmType type,
					  const std::shared_ptr<Landscape::Params> &landscapeParams,
					  const std::shared_ptr<Napvig::Params> &napvigParams);
};

using Policy = PolicyAbstract<NapvigPredictive::Params>;

/***************
 * Go straight with the direction computed by Napvig core
 * *************/

class SearchStraightPolicy : public virtual Policy
{
public:
	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getNextSearch (const Napvig::Trajectory &trajectory);
};

/***************
 * Define collision or max window termination condition
 * *************/

class CollisionTerminatedPolicy : public virtual Policy
{

public:
	virtual Termination terminationCondition (const Napvig::Trajectory &trajectory);
};

/***************
 * Follow napvig core directions until collision or max window reached
 * ************/

class StartDrivenPolicy : public SearchStraightPolicy, public CollisionTerminatedPolicy
{
public:
	StartDrivenPolicy (const std::shared_ptr<const Landscape> &_landscape,
					   const std::shared_ptr<const NapvigPredictive::Params> &_params);
};

#endif // NAPVIG_PREDICTIVE_H
