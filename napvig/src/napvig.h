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

	class Trajectory
	{
		std::vector<State> states;

	public:
		Trajectory (const std::vector<State> &_states):
			states(_states)
		{}
		Trajectory (const Trajectory &_other) = default;

		std::vector<State>::iterator begin ();
		std::vector<State>::iterator end ();
		State &operator[] (int i) {
			return states[i];
		}
		State operator[] (int i) const {
			return states[i];
		}

		DEF_SHARED(Trajectory)
	};

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
		DEF_SHARED(Params)
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

public:
	// Track openloop odometry frame update
	class FramesTracker {
		ReadyFlags<std::string> flags;
		Frame odomFrame, measuresFrame, worldFrame;

	public:
		FramesTracker ();
		void resetFrame ();
		void updateFrame (const Frame &newFrame);
		bool isReady () const;
		Frame world () const;
		Frame current () const;
		State toMeasuresFrame (const State &stateOdom) const;


		DEF_SHARED(FramesTracker)
	} framesTracker;

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

	DEF_SHARED(Napvig)
};

class ExplorativeCost;

struct NapvigDebug
{
	const std::shared_ptr<Landscape> landscapePtr;
	const std::shared_ptr<Napvig::FramesTracker> framesTrackerPtr;
	std::shared_ptr<ExplorativeCost> explorativeCostPtr;
	std::vector<float> values;
	torch::Tensor debugTensor;

	struct SearchHistory {
		struct PathTrial {
			torch::Tensor path;
			torch::Tensor initialSearch;
			double costValue;
		};
		std::vector<PathTrial> trials;
		int chosen;

		void reset ();
		void add (const Napvig::Trajectory &path, boost::optional<double> costValue = boost::none);
	} history;

	NapvigDebug (const std::shared_ptr<Landscape> &_landscape, const std::shared_ptr<Napvig::FramesTracker> &_framesTrackerPtr);

	DEF_SHARED(NapvigDebug)
};



#endif // NAPVIG_H














