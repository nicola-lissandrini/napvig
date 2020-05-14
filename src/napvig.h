#ifndef NAPVIG_H
#define NAPVIG_H

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <torch/autograd.h>
#include <ATen/Layout.h>
#include <chrono>
#define INCLUDE_TORCH
#include "common.h"

#include "napvig_map.h"

struct NapvigParams {
	double stepAheadSize;
	double gradientStepSize;
	double terminationDistance;
	int terminationCount;
};

class Napvig
{
	NapvigParams params;
	NapvigMap map;
	ReadyFlags<std::string> flags;

public:
	struct {
		torch::Tensor position;
		torch::Tensor bearing;
		torch::Tensor gradLog;
	} state;

private:
	torch::Tensor valleySearch (const torch::Tensor &xStep, const torch::Tensor &rSearch, int &num);
	torch::Tensor stepAhead (const torch::Tensor &xStart, const torch::Tensor &rSearch) const;
	torch::Tensor getBaseOrthogonal (const torch::Tensor &base) const;
	torch::Tensor projectOnto (const torch::Tensor &space, const at::Tensor &vector) const;
	torch::Tensor predict (const torch::Tensor &xStart, const torch::Tensor &rSearch, int count) const;
	torch::Tensor computeBearing (const torch::Tensor &nextPosition) const;
	torch::Tensor nextSample () ;

public:
	Napvig (const NapvigMapParams &mapParams,
			const NapvigParams &napvigParams);

	void step ();
	void setMeasures (const torch::Tensor &measures);

	double mapValue (const torch::Tensor &x) const;
	inline torch::Tensor mapGrad (const torch::Tensor &x) const {
		return map.grad (x);
	}
	inline torch::Tensor getPosition () const {
		return state.position;
	}
	inline torch::Tensor getBearing () const;
	inline void resetState ();

	bool isMapReady () const;
	bool isReady () const;
};

#endif // NAPVIG_H














