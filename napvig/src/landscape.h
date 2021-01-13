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



class Landscape
{
public:
	struct Params {
		double measureRadius;
		double smoothRadius;
		double minDistance;
		int precision;
		int dim;

		DEF_SHARED(Params)
	};

private:
	std::shared_ptr<Params> paramsData;
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
	double gammaDistance (double distance) const;

	const Params &params () const {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}
public:
	Landscape (const std::shared_ptr<Params> &_params);

	torch::Tensor value (const torch::Tensor &x) const;
	torch::Tensor grad (const torch::Tensor &x) const;
	bool collides (const torch::Tensor &x) const;
	bool isReady () const;
	int getDim () const;

	void setMeasures (const torch::Tensor &newMeasures);

	~Landscape () {
		ROS_ERROR ("LANDSCAPE DESTROYED");
	}

	DEF_SHARED(Landscape)
};

template<class T>
torch::Tensor montecarlo (torch::Tensor (T::*f)(const torch::Tensor &) const,
						  const T *obj,
						  int dim,
						  int count,
						  const torch::Tensor &center,
						  double variance,
						  bool debug) {
	torch::Tensor xVar = at::normal (0.0,variance,{count,dim});
	torch::Tensor xEval = xVar + center.expand ({count, dim});

	return (obj->*f) (xEval).mean (0);
}


#endif // NAPVIG_MAP_H
