#ifndef NAPVIG_X_H
#define NAPVIG_X_H

#include "napvig_predictive.h"

/**********
 * Define NapvigX policies
 * ********/

class FullyExploitative : public CollisionTerminatedPolicy
{
protected:
	// todo
public:
	FullyExploitative ();
};

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

class NapvigX : public NapvigPredictive
{
public:
	struct Params : Napvig::Params {
		int lookaheadHorizon;
		Range angleSearch;
	};

private:
	// Allow params inheritance
	const Params &params () const {
		return *dynamic_pointer_cast<Params> (paramsData);
	}

protected:
	std::shared_ptr<FullyExploitative> fextPolicy;
	std::shared_ptr<FullyExplorative> fexprPolicy;
	std::shared_ptr<PartiallyExploitative> pextPolicy;

	boost::optional<Napvig::Trajectory> trajectoryAlgorithm (const State &initialState);

public:
	NapvigX (const std::shared_ptr<Landscape::Params> &landscapeParams,
			 const std::shared_ptr<NapvigX::Params> &params);
};

#endif // NAPVIG_X_H
