#ifndef POLICY_H
#define POLICY_H

#include "../napvig.h"

class Policy
{
protected:
	const std::shared_ptr<const Landscape &> landscape;
	ReadyFlags<std::string> flags;

public:
	enum Termination {
		PREDICTION_TERMINATION_NONE,
		PREDICTION_TERMINATION_COLLISION,
		PREDICTION_TERMINATION_MAX_STEP,
		PREDICTION_TERMINATION_TARGET_REACHED
	};

	Policy (const std::shared_ptr<Landscape> &_landscape);

	virtual void init () {}
	virtual boost::optional<torch::Tensor> getFirstSearch (const Napvig::State &initialState) = 0;
	virtual torch::Tensor getNextSearch (const Napvig::Trajectory &trajectory) = 0;
	virtual Termination terminationCondition (const Napvig::Trajectory &trajectory) = 0;
	virtual bool selectTrajectory (const Napvig::Trajectory &trajectory, Termination termination) = 0;
	bool noTrajectory () {
		return flags["no_trajectory"];
	}
};

/***************
 * Go straight with the direction computed by Napvig core
 * *************/

class SearchStraightPolicy : public virtual Policy
{
public:
	torch::Tensor getNextSearch (const Napvig::Trajectory &trajectory);
};

/***************
 * Define collision or max window termination condition
 * *************/

class CollisionTerminatedPolicy : public virtual Policy
{
public:
	struct Params {
		int windowLength;
	};
	
private:
	Params params;
	
public:
	CollisionTerminatedPolicy (const Params &_params);
	Termination terminationCondition (const Napvig::Trajectory &trajectory);
};

/***************
 * Follow napvig core directions until collision or max window reached
 * ************/

class StartDrivenPolicy : public SearchStraightPolicy, public CollisionTerminatedPolicy
{
public:
	StartDrivenPolicy (int _windowLength);
};

#endif // POLICY_H
