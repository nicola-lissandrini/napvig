#ifndef NAPVIG_X_H
#define NAPVIG_X_H

#include "napvig_predictive.h"

class NapvigX : public NapvigPredictive
{
public:
	struct Params {
		NapvigPredictive::Params predictiveParams;
		Range angleSearch;
	};
private:
	Params params;


public:
	NapvigX();
};

#endif // NAPVIG_X_H
