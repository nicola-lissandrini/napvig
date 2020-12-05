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
	};

private:
	const Params &params() const {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}

protected:
	std::shared_ptr<RandomizePolicy> randomizePolicy;

	boost::optional<Napvig::Trajectory> trajectoryAlgorithm (const State &initialState);

public:
	NapvigRandomized (const std::shared_ptr<Landscape::Params> &_landscapeParams,
					  const std::shared_ptr<Params> &_params);
};

class RandomizePolicy : public SearchStraightPolicy, public CollisionTerminatedPolicy
{
	torch::Tensor lastSearch;
	int trials;
	bool first;

	torch::Tensor randomize (const torch::Tensor &search);

	const NapvigRandomized::Params &params() {
		return *std::dynamic_pointer_cast<const NapvigRandomized::Params> (paramsData);
	}
public:
	RandomizePolicy (const std::shared_ptr<Landscape> _landscape,
					 const std::shared_ptr<NapvigRandomized::Params> &_params);

	void init ();
	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getFirstSearch (const Napvig::State &initialState);
	bool processTrajectory (const Napvig::Trajectory &trajectory, Termination termination);
};

#endif // NAPVIG_RANDOMIZED_H
