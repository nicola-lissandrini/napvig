#ifndef ROTATION_H
#define ROTATION_H

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <iostream>
#include <eigen3/Eigen/Geometry>

#define QUATERNION_N 4
#define DIM_2D 2
#define DIM_3D 3

// Valid for both 3d and 2d rotations
// 2D rotation is a 3d rotation about axis z
class Rotation
{
	Eigen::Quaterniond quaternion;

	bool checkValid(const torch::Tensor &quaternionTensor);

	Rotation (const Eigen::Quaterniond &otherQuaternion);

public:
	Rotation ();
	Rotation (const Rotation &other) = default;
	// Tensor with order [x, y, z, w (real)], ROS convention
	Rotation (const torch::Tensor &tensorQuaternion);

	static Rotation fromAxisAngle (const torch::Tensor &axis, double angle);
	// Implicit 2d rotation around z
	static Rotation fromAxisAngle (double angle);

	Rotation operator * (const Rotation &other) const;
	torch::Tensor operator * (const torch::Tensor &vector) const;
	
	Rotation inv () const;

	friend std::ostream &operator<< (std::ostream &os, const Rotation &dt);
};

struct Frame
{
	Rotation orientation;
	torch::Tensor position;

	Frame ():
		position(torch::zeros ({2}, torch::kDouble))
	{}

	Frame (Rotation otherRotation, const torch::Tensor &otherPosition) {
		orientation = otherRotation;
		position = otherPosition;
	}

	Frame clone () const {
		return Frame(orientation, position.clone ());
	}

	Frame inv () const {
		return Frame (orientation.inv (), -(orientation.inv () * position));
	}

	torch::Tensor operator * (const torch::Tensor &vector) const {
		return orientation * vector + position;
	}

	const Frame operator * (const Frame &second) const {
		return Frame (this->orientation * second.orientation,
					  this->orientation * second.position + this->position);
	}
};

std::ostream& operator<<(std::ostream &os, const Rotation &dt);

#endif // ROTATION_H
