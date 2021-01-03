#ifndef NAPVIG_X_H
#define NAPVIG_X_H


#include "../napvig_predictive.h"
#include "landmarks.h"

/**********
 * Define NapvigX policies
 * ********/

class FullyExploitative;
class CostOptimizationPolicy;
class CostFunction;

class TargetTracker
{
	ReadyFlags<std::string> flags;

protected:
	Frame targetFrame;

public:
	TargetTracker ();

	void updateTarget (const Frame &_targetFrame);
	bool targetSet () const;
};

class NapvigX : public NapvigPredictive, public TargetTracker
{
public:
	struct Params : NapvigPredictive::Params {
		double stepGainSaturationDistance;
		double targetReachedThreshold;
		struct LandmarkParams {
			int maxQueue;
			double minimumDistanceCreation;
			double maximumTimeCreation;
			double forgettingFactor;
			double radius;
		} landmarks;
		Range angleSearch;
	};

private:
	LandmarksBatch landmarks;

	// Allow params inheritance
	const Params &params () const {
		return *std::dynamic_pointer_cast<Params> (paramsData);
	}

	bool checkTargetUnreachable () const;
	void updateLandmarks ();
	bool checkLandmarkNew (const Landmark &current, const Landmark &last);

protected:
	std::shared_ptr<FullyExploitative> exploitativePolicy;
	std::shared_ptr<CostOptimizationPolicy> costOptimizationPolicy;

	std::shared_ptr<CostFunction> fullyExplorativeCost;

	boost::optional<Napvig::Trajectory> trajectoryAlgorithm (const State &initialState);

public:
	NapvigX (const std::shared_ptr<Landscape::Params> &landscapeParams,
			 const std::shared_ptr<NapvigX::Params> &params);
};

class FullyExploitative : public CollisionTerminatedPolicy
{
	std::shared_ptr<Frame> targetFrame;

	const NapvigX::Params &params () const {
		return *std::dynamic_pointer_cast<const NapvigX::Params> (paramsData);
	}

	std::pair<torch::Tensor,boost::optional<torch::Tensor>> searchTowardsTarget (const torch::Tensor &currentPosition);
	bool checkTargetReached () const;

public:
	FullyExploitative(const std::shared_ptr<Landscape> &_landscape,
					  const std::shared_ptr<NapvigX::Params> &_params,
					  const std::shared_ptr<Frame> &_targetFrame);

	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getFirstSearch (const Napvig::State &initialState);
	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getNextSearch (const Napvig::Trajectory &trajectory);
	bool processTrajectory (const Napvig::Trajectory &trajectory, Termination termination);
	Termination terminationCondition(const Napvig::Trajectory &trajectory);
};

class CostFunction
{
protected:
	std::shared_ptr<NapvigPredictive::Params> paramsData;

public:
	CostFunction (const std::shared_ptr<NapvigPredictive::Params> &_paramsData);

	virtual double get (const Napvig::Trajectory &trajectory) = 0;
};

// A cost based on landmarks
class ExplorativeCost : public CostFunction
{
protected:
	const NapvigX::Params &params () const {
		return *std::dynamic_pointer_cast<NapvigX::Params> (paramsData);
	}

protected:
	std::shared_ptr<LandmarksBatch> landmarks;
	std::shared_ptr<Napvig::FramesTracker> framesTracker;

	virtual double pointCost (const torch::Tensor &point) = 0;
	void convertLandmarks ();

	friend class NapvigNodeDebugger;

public:
	ExplorativeCost (const std::shared_ptr<NapvigPredictive::Params> &_params,
					 const std::shared_ptr<LandmarksBatch> &_landmarks,
					 const std::shared_ptr<Napvig::FramesTracker> &_framesTracker);

	double get (const Napvig::Trajectory &trajectory) final;
};

class FullyExplorativeCost : public ExplorativeCost
{
	double landmarkCostAt (const torch::Tensor &x,
						   const torch::Tensor &landmarkPosition);

public:
	double pointCost (const torch::Tensor &point);

public:
	FullyExplorativeCost (const std::shared_ptr<NapvigPredictive::Params> &_params,
						  const std::shared_ptr<LandmarksBatch> &_landmarks,
						  const std::shared_ptr<Napvig::FramesTracker> &_framesTracker);

};

// Calls sequence:
// init
// newFirst = next
// newTraj = [generate with newFirst]
// add (newTraj)

class IterativeSampledOptimizer
{
protected:
	std::shared_ptr<CostFunction> cost;
	torch::Tensor trajectoriesCosts;
	std::vector<std::shared_ptr<Napvig::Trajectory>> trajectories;
	int current;

	virtual void reset () {}

public:
	IterativeSampledOptimizer (const std::shared_ptr<CostFunction> &_cost, int samples);

	void init ();
	// Receive previous and gives next initial direction
	virtual torch::Tensor next () = 0;
	virtual bool isLast () = 0;
	void add (const std::shared_ptr<Napvig::Trajectory> &trajectory);
	std::pair<std::shared_ptr<Napvig::Trajectory>, int> getOpt ();
};

class SampledAngleOptimizer : public virtual IterativeSampledOptimizer
{
protected:
	torch::Tensor initialSearch;
	torch::Tensor getDirection (double angle);
	
public:
	void setInitialSearch (const torch::Tensor &_initialSearch);
};

class GridAngleOptimizer : public SampledAngleOptimizer
{
	Range searchRange;
	double angle;

	void reset ();

public:
	GridAngleOptimizer (const std::shared_ptr<CostFunction> &_cost, const Range &_searchRange):
		IterativeSampledOptimizer(_cost, _searchRange.count ()),
		searchRange(_searchRange)
	{}
	
	torch::Tensor next ();
	bool isLast ();
};

class CostOptimizationPolicy : public SearchStraightPolicy, public CollisionTerminatedPolicy
{

	std::shared_ptr<CostFunction> cost;
	std::shared_ptr<GridAngleOptimizer> optimizer;

	const NapvigX::Params &params () const {
		return *std::dynamic_pointer_cast<const NapvigX::Params> (paramsData);
	}

	void init();
public:
	CostOptimizationPolicy (const std::shared_ptr<Landscape> &_landscape,
							const std::shared_ptr<NapvigX::Params> &_params,
							const std::shared_ptr<CostFunction> &_cost);

	void setCost (const std::shared_ptr<CostFunction> &_cost);

	std::pair<torch::Tensor,boost::optional<torch::Tensor>> getFirstSearch (const Napvig::State &initialState);
	bool processTrajectory (const Napvig::Trajectory &trajectory, Termination termination);

};


/*
class PartiallyExploitative : public StartDrivenPolicy, public CollisionTerminatedPolicy
{
public:
	PartiallyExploitative ();
};
*/

#endif // NAPVIG_X_H
