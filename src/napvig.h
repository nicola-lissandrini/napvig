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

struct NapvigDebug
{
	std::vector<double> values;
	torch::Tensor corridor;
};

class Napvig
{
public:
	struct State {
		torch::Tensor position;
		torch::Tensor search;
		bool randomized;

		State clone () {
			return State {position.clone (), search.clone()};
		}
	};

	struct SearchHistory {
		std::vector<torch::Tensor> triedPaths;
		std::vector<torch::Tensor> initialSearches;
		int chosen;
	};

	enum AlgorithmType {
		SINGLE_STEP,
		RANDOMIZED_RECOVERY,
		OPTIMIZED_TRAJECTORY
	};

	enum NapvigMode {
		EXPLORATION,
		EXPLOITATION
	};

	struct Params {
		double stepAheadSize;
		double gradientStepSize;
		double terminationDistance;
		double minDistance;
		double scatterVariance;
		int lookaheadHorizon;
		int terminationCount;
		bool keepLastSearch;
		AlgorithmType algorithm;

		struct {
			double rangeAngleMin;
			double rangeAngleMax;
			double rangeAngleStep;
		} trajectoryOptimizerParams;
	};

private:
	NapvigMap map;
	NapvigDebug *debug;
	Params params;
	SearchHistory lastHistory;
	ReadyFlags<std::string> flags, stateFlags;
	const torch::Tensor initialPosition = torch::zeros ({2}, torch::kDouble);
	const torch::Tensor initialSearch = torch::tensor ({1.0, 0.0}, torch::kDouble);
	
	State state, oldState, setpoint;
	Frame frame, oldFrame;
	Frame targetFrame;
	bool targetUnreachable;
	NapvigMode mode;

	torch::Tensor valleySearch (const torch::Tensor &xStep, const torch::Tensor &rSearch, int &num) const;
	State nextSample (const State &q) const;
	State step (State &q) const;
	torch::Tensor stepAhead (const State &q) const;
	torch::Tensor getBaseOrthogonal (const torch::Tensor &base) const;
	torch::Tensor projectOnto (const torch::Tensor &space, const at::Tensor &vector) const;
	State randomize (const State &state) const;
	void keepLastSearch ();
	std::tuple<State, bool, torch::Tensor> predictTrajectory(const State &initialState, int maxCount) const;
	torch::Tensor computeBearing (const torch::Tensor &oldPosition, const torch::Tensor &nextPosition) const;
	double evaluateCost (const torch::Tensor &trajectory) const;
	double costDistanceToTarget (const torch::Tensor &trajectory) const;
	double costPathLength (const torch::Tensor &trajectory) const;

	bool collides (const torch::Tensor &x) const;
	double gammaDistance (double distance) const;

public:
	Napvig (const NapvigMap::Params &mapParams,
			const Params &napvigParams,
			NapvigDebug *_debug);

	State stepSingle (State initialState);
	std::pair<State, SearchHistory> stepRandomizedRecovery(State initialState);
	std::pair<State, SearchHistory> stepOptimizedTrajectory (State initialState);
	bool step();

	void setMeasures (const torch::Tensor &measures);
	void setCorridor (const torch::Tensor &corridor);

	double mapValue (const torch::Tensor &x) const;
	torch::Tensor mapGrad (const torch::Tensor &x) const;
	torch::Tensor getSetpointPosition () const;
	torch::Tensor getSetpointDirection () const;
	SearchHistory getSearchHistory () const;
	void updatePosition ();
	void updateOrientation ();
	void updateFrame (Frame newFrame);
	void updateTarget (Frame newTargetFrame);
	double getDistanceFromCorridor ();

	bool isMapReady () const;
	bool isReady () const;
	bool isTargetUnreachable () const;

	int getDim () const;
};

#endif // NAPVIG_H














