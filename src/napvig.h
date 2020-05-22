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

enum NapvigAlgorithmType {
	NAPVIG_SINGLE_STEP,
	NAPVIG_PREDICT_COLLISION
};

#include <complex>

typedef std::complex<double> complexd;

struct NapvigParams {
	double stepAheadSize;
	double gradientStepSize;
	double terminationDistance;
	double minDistance;
	double scatterVariance;
	int lookaheadHorizon;
	int terminationCount;
	NapvigAlgorithmType algorithm;
};

struct SearchHistory {
	std::vector<torch::Tensor> triedPaths;
	std::vector<torch::Tensor> initialSearches;
};

class Napvig
{
	NapvigMap map;
	NapvigParams params;
	ReadyFlags<std::string> flags;
	const torch::Tensor initialPosition = torch::zeros ({2}, torch::kDouble);
	const torch::Tensor initialSearch = torch::tensor ({1.0, 0.0}, torch::kDouble);

public:
	torch::Tensor gradLog;
	struct State {
		torch::Tensor position;
		torch::Tensor search;
	} state, oldState;

	complexd frame, oldFrame;

private:
	std::pair<torch::Tensor, torch::Tensor> valleySearch (const torch::Tensor &xStep, const torch::Tensor &rSearch, int &num) const;
	std::pair<State, torch::Tensor> nextSample (const State &q) const;
	torch::Tensor stepAhead (const State &q) const;
	torch::Tensor getBaseOrthogonal (const torch::Tensor &base) const;
	torch::Tensor projectOnto (const torch::Tensor &space, const at::Tensor &vector) const;
	State randomize (const State &state) const;
	std::tuple<State, bool, torch::Tensor> predictCollision(const State &initialState, int maxCount) const;
	torch::Tensor computeBearing (const torch::Tensor &oldPosition, const torch::Tensor &nextPosition) const;

	bool collides (const torch::Tensor &x) const;
	double gammaDistance (double distance) const;

public:
	Napvig (const NapvigMapParams &mapParams,
			const NapvigParams &napvigParams);

	State stepSingle ();
	std::pair<State, SearchHistory> stepDiscovery();
	void step ();

	void setMeasures (const torch::Tensor &measures);

	double mapValue (const torch::Tensor &x) const;
	torch::Tensor mapGrad (const torch::Tensor &x) const;
	torch::Tensor getPosition () const;
	torch::Tensor getBearing () const;
	void resetState ();
	void updateFrame (complexd newFrame);

	bool isMapReady () const;
	bool isReady () const;
};

#endif // NAPVIG_H














