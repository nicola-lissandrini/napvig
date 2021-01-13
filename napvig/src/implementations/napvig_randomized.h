#ifndef NAPVIG_RANDOMIZED_H
#define NAPVIG_RANDOMIZED_H

#include "napvig_predictive.h"

class RandomizePolicy;

class NapvigRandomized : public NapvigPredictive
{
public:
	struct Params : NapvigPredictive::Params {
		double randomizeVariance;
		int maxTrials;
		DEF_SHARED(Params)
	};

private:
	const Params &params() const {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}

protected:
	RandomizePolicy::Ptr randomizePolicy;

	boost::optional<Napvig::Trajectory> trajectoryAlgorithm (const State &initialState);

public:
	NapvigRandomized (const Landscape::Params::Ptr &_landscapeParams,
					  const NapvigRandomized::Params::Ptr &_params);

	DEF_SHARED(NapvigRandomized)
};

class RandomizePolicy : public SearchStraightPolicy, public CollisionTerminatedPolicy
{
	torch::Tensor lastSearch;
	int trials, index;
	bool first;

	torch::Tensor randomize (const torch::Tensor &search);

	const NapvigRandomized::Params &params() {
		return *std::dynamic_pointer_cast<const NapvigRandomized::Params> (paramsData);
	}
public:
	RandomizePolicy (const Landscape::Ptr _landscape,
					 const NapvigRandomized::Params::Ptr &_params);

	void init ();
	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getFirstSearch (const Napvig::State &initialState);
	bool processTrajectory (const Napvig::Trajectory &trajectory, Termination termination);

	DEF_SHARED(RandomizePolicy)
};

#endif // NAPVIG_RANDOMIZED_H
