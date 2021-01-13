#include "napvig_x.h"

using namespace std;
using namespace torch;

TargetTracker::TargetTracker() {
	flags.addFlag ("first_target");
}

void TargetTracker::updateTarget(const Frame &_targetFrame) {
	flags.set("first_target");
	targetFrame = _targetFrame;
}

bool TargetTracker::targetSet() const {
	return flags["first_target"];
}


NapvigX::NapvigX (const std::shared_ptr<Landscape::Params> &_landscapeParams,
				  const std::shared_ptr<NapvigX::Params> &_params):
	NapvigPredictive(NAPVIG_X,
					 _landscapeParams,
					 _params)
{

	fullyExplorativeCost = make_shared<FullyExplorativeCost> (_params,
															  shared_ptr<LandmarksBatch>(&landmarks),
															  shared_ptr<FramesTracker>(&framesTracker));

	partiallyExplorativeCost = make_shared<PartiallyExplorativeCost> (_params,
																	  shared_ptr<LandmarksBatch>(&landmarks),
																	  shared_ptr<FramesTracker>(&framesTracker),
																	  shared_ptr<Frame> (&targetFrame));

	exploitativePolicy = make_shared<FullyExploitative> (shared_ptr<Landscape>(&landscape),
														 _params,
														 shared_ptr<Frame> (&targetFrame));

	costOptimizationPolicy = make_shared<CostOptimizationPolicy> (shared_ptr<Landscape>(&landscape),
																  _params,
																  partiallyExplorativeCost);

	debug->explorativeCostPtr = dynamic_pointer_cast<ExplorativeCost> (partiallyExplorativeCost);
}

bool NapvigX::checkTargetUnreachable() const {
	return landscape.collides (targetFrame.position);
}

void NapvigX::updateLandmarks()
{
	Landmark newLandmark = Landmark (framesTracker.current ());

	ROS_INFO ("Queue %d", landmarks.size ());

	// Empty queue
	if (landmarks.size () == 0) {
		landmarks.push (newLandmark);
		return;
	}

	// Overfull queue
	if (landmarks.size () > params().landmarks.maxQueue)
		landmarks.pop ();

	const Landmark &last = landmarks.last ();

	// Check whether the new landmark should be stored
	if (checkLandmarkNew (newLandmark, last)) {
		ROS_INFO ("New landmark!!!! size %d", landmarks.size ());
		landmarks.push (newLandmark);
	}
}

bool NapvigX::checkLandmarkNew(const Landmark &current, const Landmark &last)
{
	const Frame currentFrame = framesTracker.current ();
	if ((current.getInFrame (currentFrame) - last.getInFrame (currentFrame)).norm ().item().toDouble () > params().landmarks.minimumDistanceCreation) {
		ROS_INFO ("too far");
		return true;
	}
	if (last.elapsed () > params().landmarks.maximumTimeCreation) {
		ROS_INFO ("too old");
		return true;
	}

	return false;
}

boost::optional<Napvig::Trajectory> NapvigX::trajectoryAlgorithm (const State &initialState)
{
	updateLandmarks ();
/*
	if (!targetSet ())
		return boost::none;

	if (checkTargetUnreachable ()) {
		ROS_WARN ("Target unreachable");
		return boost::none;
	} else {
		ROS_INFO ("++TARGET REACHABLE");
	}

	// FUTURE: Check an existing Fully Exploitative trajectory exists

	// Fully exploitative
	//return followPolicy (initialState, exploitativePolicy);
*/
	// Explorative
	costOptimizationPolicy->setCost (fullyExplorativeCost);

	return followPolicy (initialState, costOptimizationPolicy);
}























































