#ifndef NAPVIG_X_H
#define NAPVIG_X_H

#include "napvig_predictive.h"

/**********
 * Define NapvigX policies
 * ********/

class FullyExploitative;
class FullyExplorative;
class PartiallyExploitative;

class TargetTracker
{
	ReadyFlags<std::string> flags;

protected:
	Frame targetFrame;

public:
	TargetTracker ();

	void updateTarget (const Frame &_targetFrame);
	bool targetSet () const;
};

class NapvigX : public NapvigPredictive, public TargetTracker
{
public:
	struct Params : NapvigPredictive::Params {
		double stepGainSaturation;
		double targetReachedThreshold;
		Range angleSearch;
	};

private:
	// Allow params inheritance
	const Params &params () const {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}

protected:
	std::shared_ptr<FullyExploitative> fextPolicy;
/*	std::shared_ptr<FullyExplorative> fexprPolicy;
	std::shared_ptr<PartiallyExploitative> pextPolicy;*/

	boost::optional<Napvig::Trajectory> trajectoryAlgorithm (const State &initialState);

public:
	NapvigX (const std::shared_ptr<Landscape::Params> &landscapeParams,
			 const std::shared_ptr<NapvigX::Params> &params);
};

class FullyExploitative : public CollisionTerminatedPolicy
{
	std::shared_ptr<Frame> targetFrame;

	const NapvigX::Params &params () const {
		return *std::dynamic_pointer_cast<const NapvigX::Params> (paramsData);
	}

	std::pair<torch::Tensor,boost::optional<torch::Tensor>> searchTowardsTarget (const torch::Tensor &currentPosition);
	bool checkTargetReached () const;

public:
	FullyExploitative(const std::shared_ptr<Landscape> &_landscape,
					  const std::shared_ptr<NapvigPredictive::Params> &_params,
					  const std::shared_ptr<Frame> &_targetFrame);

	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getFirstSearch (const Napvig::State &initialState);
	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getNextSearch (const Napvig::Trajectory &trajectory);
	bool processTrajectory (const Napvig::Trajectory &trajectory, Termination termination);
	Termination terminationCondition(const Napvig::Trajectory &trajectory);
};
/*
class FullyExplorative : public StartDrivenPolicy, public CollisionTerminatedPolicy
{
public:
	FullyExplorative ();
};

class PartiallyExploitative : public StartDrivenPolicy, public CollisionTerminatedPolicy
{
public:
	PartiallyExploitative ();
};
*/

#endif // NAPVIG_X_H
