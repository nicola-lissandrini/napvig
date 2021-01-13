#ifndef NAPVIG_PREDICTIVE_H
#define NAPVIG_PREDICTIVE_H

#include "../napvig.h"
#include "policy.h"


class NapvigPredictive : public Napvig
{
public:
	struct Params : Napvig::Params {
		int windowLength;
		DEF_SHARED(Params)
	};


protected:
	std::pair<Napvig::Trajectory, PolicyAbstract<NapvigPredictive::Params>::Termination> predictTrajectory (const State &initialState, const PolicyAbstract<NapvigPredictive::Params>::Ptr &policy);
	boost::optional<Napvig::Trajectory> followPolicy (const State &initialState, const PolicyAbstract<NapvigPredictive::Params>::Ptr &policy);

public:
	NapvigPredictive (AlgorithmType type,
					  const Landscape::Params::Ptr &landscapeParams,
					  const Params::Ptr &napvigParams);

	DEF_SHARED(NapvigPredictive)
};

using Policy = PolicyAbstract<NapvigPredictive::Params>;

/***************
 * Go straight with the direction computed by Napvig core
 * *************/

class SearchStraightPolicy : public virtual Policy
{
public:
	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getNextSearch (const Napvig::Trajectory &trajectory);

	DEF_SHARED(SearchStraightPolicy)
};

/***************
 * Define collision or max window termination condition
 * *************/

class CollisionTerminatedPolicy : public virtual Policy
{

public:
	virtual Termination terminationCondition (const Napvig::Trajectory &trajectory);

	DEF_SHARED(CollisionTerminatedPolicy)
};


#endif // NAPVIG_PREDICTIVE_H
