#ifndef NAPVIG_X_H
#define NAPVIG_X_H

#include "napvig_predictive.h"

class NapvigX : public NapvigPredictive
{
public:
	struct Params : Napvig::Params {
		int lookaheadHorizon;
		Range angleSearch;
	};
protected:
	// Allow params inheritance
	const Params &params () {
		return *dynamic_pointer_cast<Params> (paramsData);
	}

public:
	NapvigX (const std::shared_ptr<Landscape::Params> &landscapeParams,
			 const std::shared_ptr<NapvigX::Params> &params);
};

#endif // NAPVIG_X_H
