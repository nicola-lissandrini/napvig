#ifndef POLICY_H
#define POLICY_H

#include "../napvig.h"

// Workaround for NapvigPredictive::Params forward declaration
template<class ParamsAbstract>
class PolicyAbstract
{

protected:
	std::shared_ptr<ParamsAbstract> paramsData;
	const std::shared_ptr<Landscape> landscape;
	ReadyFlags<std::string> flags;

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
		landscape(_landscape),
		paramsData(_params)
	{
		flags.addFlag ("no_trajectory");
	}

	virtual void init () {}
	virtual torch::Tensor getFirstSearch (const Napvig::State &initialState) = 0;
	virtual torch::Tensor getNextSearch (const Napvig::Trajectory &trajectory) = 0;
	virtual Termination terminationCondition (const Napvig::Trajectory &trajectory) = 0;
	virtual bool selectTrajectory (const Napvig::Trajectory &trajectory, Termination termination) = 0;
	bool noTrajectory () {
		return flags["no_trajectory"];
	}
};


#endif // POLICY_H
