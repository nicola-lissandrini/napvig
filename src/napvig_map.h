#ifndef NAPVIG_MAP_H
#define NAPVIG_MAP_H

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <torch/autograd.h>
#include <ATen/Layout.h>
#include <chrono>
#define INCLUDE_TORCH
#include "common.h"

#define N_DIM 2



class NapvigMap
{
public:
	struct Params {
		double measureRadius;
		double smoothRadius;
		int precision;
		int dim;
	};
private:
	Params params;
	torch::Tensor measures;
	ReadyFlags<std::string> flags;
	double smoothGain;

	double getSmoothGain() const;
	double getNoAmplificationGain() const;

	torch::Tensor gaussian(const torch::Tensor &x, double sigma) const;
	torch::Tensor gamma (const torch::Tensor &x) const;
	torch::Tensor potentialFunction (const torch::Tensor &x) const;
	torch::Tensor exponentialPart (const torch::Tensor &distToMeasures, const torch::Tensor &kIndex) const;
	torch::Tensor exponentialPart (const torch::Tensor &selectedDistToMeasures) const;
	torch::Tensor preSmooth (const torch::Tensor &x) const;
	torch::Tensor preSmoothGrad (const torch::Tensor &x) const;
	torch::Tensor normSquare (const torch::Tensor &x) const;

public:
	NapvigMap (const Params &_params);

	torch::Tensor value (const torch::Tensor &x) const;
	torch::Tensor grad (const torch::Tensor &x) const;
	double gammaDistance (double distance) const;

	bool isReady () const;
	int getDim () const;

	void setMeasures (const torch::Tensor &newMeasures);
};

template<class T>
torch::Tensor montecarlo (torch::Tensor (T::*f)(const torch::Tensor &) const, const T *obj, int dim, int count, const torch::Tensor &center, double variance) {
	torch::Tensor xEval = torch::rand ({count, dim}) * variance + center.expand ({count, dim});

	return (obj->*f) (xEval).mean (0);
}


#endif // NAPVIG_MAP_H
