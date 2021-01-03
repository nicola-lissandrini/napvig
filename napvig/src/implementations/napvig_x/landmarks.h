#ifndef LANDMARKS_H
#define LANDMARKS_H

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>
#include <chrono>

#include "rotation.h"

class Landmark
{
	Frame frameAtCreation;
	torch::Tensor convertedPosition;
	std::chrono::time_point<std::chrono::steady_clock> createdTime;

public:
	explicit Landmark (const Frame &_frame);
	Landmark (const Landmark &_other) = default;

	double elapsed () const;
	void convertPosition(const Frame &current);
	torch::Tensor getWorldPosition () const;
	torch::Tensor getConvertedPosition () const;
	torch::Tensor getInFrame (const Frame &frame) const;
};

using LandmarksQueue = std::deque<Landmark>;

class LandmarksBatch
{
	LandmarksQueue landmarks;

public:
	LandmarksBatch () = default;

	void push (const Landmark &landmark);
	void pop ();
	const Landmark &last ();
	std::size_t size ();

	void convertAll (const Frame &_frame);
	LandmarksQueue::iterator begin ();
	LandmarksQueue::iterator end ();

};

#endif // LANDMARKS_H
