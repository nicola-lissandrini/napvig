#include "landmarks.h"

using namespace std;
using namespace torch;

Landmark::Landmark (const Frame &_frame):
	frameAtCreation(_frame)
{
	createdTime = chrono::steady_clock::now ();
}

void Landmark::convertPosition (const Frame &current) {
	convertedPosition = getInFrame (current);
}

Tensor Landmark::getWorldPosition() const {
	return frameAtCreation.position;
}

Tensor Landmark::getConvertedPosition() const {
	return convertedPosition;
}

Tensor Landmark::getInFrame(const Frame &frame) const {
	return (frame.inv() * frameAtCreation).position;
}

double Landmark::elapsed() const {
	chrono::duration<double> elapsed = chrono::steady_clock::now () - createdTime;
	return elapsed.count ();
}

void LandmarksBatch::push(const Landmark &landmark) {
	landmarks.push_front (landmark);
}

void LandmarksBatch::pop() {
	return landmarks.pop_back ();
}

const Landmark &LandmarksBatch::last() {
	return landmarks.front ();
}

size_t LandmarksBatch::size() {
	return landmarks.size ();
}

void LandmarksBatch::convertAll (const Frame &_frame)
{
	for (Landmark &landmark : landmarks)
		landmark.convertPosition (_frame);
}

LandmarksQueue::iterator LandmarksBatch::begin() {
	return landmarks.begin ();
}

LandmarksQueue::iterator LandmarksBatch::end() {
	return landmarks.end ();
}
