#ifndef POLICY_H
#define POLICY_H

#include "../napvig.h"

// template: Workaround for NapvigPredictive::Params forward declaration
template<class ParamsAbstract>
class PolicyAbstract
{

protected:
	std::shared_ptr<ParamsAbstract> paramsData;
	const std::shared_ptr<Landscape> landscape;
	std::pair<boost::optional<Napvig::Trajectory>, int> finalTrajectoryIndexed;

	const ParamsAbstract &params() const {
		return *std::dynamic_pointer_cast<const ParamsAbstract> (paramsData);
	}

public:
	enum Termination {
		PREDICTION_TERMINATION_NONE,
		PREDICTION_TERMINATION_COLLISION,
		PREDICTION_TERMINATION_MAX_STEP,
		PREDICTION_TERMINATION_TARGET_REACHED
	};

	PolicyAbstract (const std::shared_ptr<Landscape> &_landscape,
					const std::shared_ptr<ParamsAbstract> &_params):
		paramsData(_params),
		landscape(_landscape)
	{
	}

	virtual void init () {}
	virtual std::pair<torch::Tensor,boost::optional<torch::Tensor>> getFirstSearch (const Napvig::State &initialState) = 0;
	virtual std::pair<torch::Tensor,boost::optional<torch::Tensor>> getNextSearch (const Napvig::Trajectory &trajectory) = 0;
	virtual Termination terminationCondition (const Napvig::Trajectory &trajectory) = 0;
	virtual bool processTrajectory (const Napvig::Trajectory &trajectory, Termination termination) = 0;
	std::pair<boost::optional<Napvig::Trajectory>, int> getFinalTrajectory () {
		return finalTrajectoryIndexed;
	}

	DEF_SHARED(PolicyAbstract)
};


#endif // POLICY_H
