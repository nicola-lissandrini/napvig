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
	Rotation (const Rotation &other) {
		quaternion = other.quaternion;
	}
	// Tensor with order [x, y, z, w (real)], ROS convention
	Rotation (const torch::Tensor &tensorQuaternion);

	static Rotation fromAxisAngle (const torch::Tensor &axis, const torch::Tensor &angle);

	Rotation operator * (const Rotation &other);
	torch::Tensor operator * (const torch::Tensor &vector);
	
	Rotation inv ();

	friend std::ostream &operator<< (std::ostream &os, const Rotation &dt);
	static torch::Tensor axis2d () {
		return torch::tensor({0,0,1},torch::kDouble);
	}
};

struct Frame
{
	Rotation orientation;
	torch::Tensor position;
	Frame () {position = torch::zeros ({2}, torch::kDouble);}
	Frame (Rotation otherRotation, const torch::Tensor &otherPosition) {
		orientation = otherRotation;
		position = otherPosition;
	}
	Frame clone () {
		return Frame(orientation, position.clone ());
	}
};

std::ostream& operator<<(std::ostream &os, const Rotation &dt);

#endif // ROTATION_H
