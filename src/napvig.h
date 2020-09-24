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
#include "rotation.h"

#include <complex>

class Napvig
{
public:
	struct State {
		torch::Tensor position;
		torch::Tensor search;
		bool randomized;
	};

	struct SearchHistory {
		std::vector<torch::Tensor> triedPaths;
		std::vector<torch::Tensor> initialSearches;
	};

	enum AlgorithmType {
		SINGLE_STEP,
		PREDICT_COLLISION
	};

	struct Params {
		double stepAheadSize;
		double gradientStepSize;
		double terminationDistance;
		double minDistance;
		double scatterVariance;
		int lookaheadHorizon;
		int terminationCount;
		AlgorithmType algorithm;
	};

private:
	NapvigMap map;
	Params params;
	SearchHistory lastHistory;
	ReadyFlags<std::string> flags, stateFlags;
	const torch::Tensor initialPosition = torch::zeros ({2}, torch::kDouble);
	const torch::Tensor initialSearch = torch::tensor ({1.0, 0.0}, torch::kDouble);
	
	State state, oldState, setpoint;
	Frame frame, oldFrame;

	torch::Tensor valleySearch (const torch::Tensor &xStep, const torch::Tensor &rSearch, int &num) const;
	State nextSample (const State &q) const;
	State step (State &q) const;
	torch::Tensor stepAhead (const State &q) const;
	torch::Tensor getBaseOrthogonal (const torch::Tensor &base) const;
	torch::Tensor projectOnto (const torch::Tensor &space, const at::Tensor &vector) const;
	State randomize (const State &state) const;
	void keepLastSearch ();
	std::tuple<State, bool, torch::Tensor> predictCollision(const State &initialState, int maxCount) const;
	torch::Tensor computeBearing (const torch::Tensor &oldPosition, const torch::Tensor &nextPosition) const;

	bool collides (const torch::Tensor &x) const;
	double gammaDistance (double distance) const;

public:
	Napvig (const NapvigMap::Params &mapParams,
			const Params &napvigParams);

	State stepSingle ();
	std::pair<State, SearchHistory> stepDiscovery();
	bool step();

	void setMeasures (const torch::Tensor &measures);

	double mapValue (const torch::Tensor &x) const;
	torch::Tensor mapGrad (const torch::Tensor &x) const;
	torch::Tensor getSetpointPosition () const;
	torch::Tensor getSetpointDirection () const;
	SearchHistory getSearchHistory () const;
	void updatePosition ();
	void updateOrientation ();
	void updateFrame (Frame newFrame);
	void updateTarget (Frame targetFrame);

	bool isMapReady () const;
	bool isReady () const;

	int getDim () const;
};

#endif // NAPVIG_H














