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

#include "landscape.h"
#include "rotation.h"

#include <optional>

struct NapvigDebug;

class Napvig
{
public:
	struct State {
		torch::Tensor position;
		torch::Tensor search;
		boost::optional<torch::Tensor> stepGain;

		State clone () const {
			return State {position.clone (), search.clone()};
		}
	};

	using Trajectory = std::vector<State>;

	enum AlgorithmType {
		NAPVIG_LEGACY,			// ICRA 2021
		NAPVIG_RANDOMIZED,		// Dismissed
		NAPVIG_X,				// T-RO
		NAPVIG_CUBE,			// IROS 2021?
		NAPVIG_HYPER_DIM,		// T-RO?
		NAPVIG_FUSION,			// Mixed with RRT* offline
		NAPVIG_COLLABORATIVE	// 2025
	};

	struct Params {
		double stepAheadSize;
		double gradientStepSize;
		double terminationDistance;
		int terminationCount;

		// Allow dynamic cast
		virtual ~Params() = default;
	};

protected:
	std::shared_ptr<Params> paramsData;

	// Allow params inheritance
	const Params &params () const {
		return *paramsData;
	}

protected:
	AlgorithmType type;
	Landscape landscape;

	// Debug info
	std::shared_ptr<NapvigDebug> debug;

	// Main algorithm
	class Core {
		const Params &params;
		Landscape &landscape;

		torch::Tensor projectOnto (const torch::Tensor &space, const torch::Tensor &vector) const;
		torch::Tensor getBaseOrthogonal (const torch::Tensor &base) const;
		torch::Tensor stepAhead (const State &q) const;
		torch::Tensor valleySearch (const torch::Tensor &xStep, const torch::Tensor &rSearch) const;
		torch::Tensor nextSearch (const torch::Tensor &current, const torch::Tensor &next) const;

	public:
		Core (const Params &_params, Landscape &parentLandscape);

		State compute (const State &q) const;
	} core;

	const State zeroState = {torch::zeros ({2}, torch::kDouble), torch::tensor ({1.0, 0.0}, torch::kDouble)};

	virtual boost::optional<Trajectory> trajectoryAlgorithm (const State &initialState) = 0;

private:
	// Track openloop odometry frame update
	class FramesTracker {
		ReadyFlags<std::string> flags;
		Frame odomFrame, measuresFrame;

	public:
		FramesTracker ();
		void resetFrame ();
		void updateFrame (const Frame &newFrame);
		bool isReady () const;
		State toMeasuresFrame (const State &stateOdom);
	} framesTracker;

public:
	Napvig (AlgorithmType _type,
			const std::shared_ptr<Landscape::Params> &landscapeParams,
			const std::shared_ptr<Params> &napvigParams);
	
	boost::optional<Trajectory> computeTrajectory ();

	void setMeasures (const torch::Tensor &measures);
	void updateFrame (const Frame &newFrame);

	std::shared_ptr<NapvigDebug> getDebug ();

	bool isReady () const;
	AlgorithmType getType () const;
	torch::Tensor getZero () const;
};

struct NapvigDebug
{
	const std::shared_ptr<Landscape> landscape;
	std::vector<float> values;
	torch::Tensor debugTensor;

	struct SearchHistory {
		std::vector<torch::Tensor> triedPaths;
		std::vector<torch::Tensor> initialSearches;
		int chosen;

		void reset ();
		void add (const Napvig::Trajectory &path);
	} history;

	NapvigDebug (const std::shared_ptr<Landscape> &_landscape);
};



#endif // NAPVIG_H














