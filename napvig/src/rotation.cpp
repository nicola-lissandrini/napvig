#include "rotation.h"
#include "napvig.h"

using namespace torch;
using namespace torch::indexing;
using namespace std;
using namespace Eigen;

bool Rotation::checkValid(const Tensor &quaternionTensor) {
	if (quaternionTensor.size(0) != QUATERNION_N)
		return false;

	return true;
}

Rotation::Rotation ():
	quaternion(1,0,0,0)
{
}

Rotation::Rotation (const Quaterniond &otherQuaternion):
	quaternion(otherQuaternion)
{}

Rotation::Rotation (const Tensor &tensorQuaternion)
{
	assert (checkValid (tensorQuaternion) && "Supplied tensor is not a valid quaternion");

	quaternion = Quaterniond (tensorQuaternion[3].item().toDouble (),
			tensorQuaternion[0].item().toDouble (),
			tensorQuaternion[1].item().toDouble (),
			tensorQuaternion[2].item().toDouble ());
}

Rotation Rotation::inv() const {
	return Rotation (quaternion.inverse ());
}

VectorXd torchVectorToEigen (const Tensor &torchVector) {
	VectorXd eigenVector;

	eigenVector.resize (torchVector.size(0));
	for (int i= 0; i < torchVector.size(0); i++)
		eigenVector[i] = torchVector[i].item ().toDouble ();

	return eigenVector;
}

Tensor eigenVectorToTorch (const VectorXd &eigenVector) {
	return torch::from_blob ((void *)eigenVector.data (),{(int)eigenVector.size ()},kDouble).clone ();
}

Rotation Rotation::fromAxisAngle (const Tensor &axis, double angle) {
	return Rotation (Quaterniond (
				cos(angle/2),
				sin(angle/2) * axis[0].item().toDouble (),
			sin(angle/2) * axis[1].item().toDouble (),
			sin(angle/2) * axis[2].item().toDouble ()));
}

Rotation Rotation::fromAxisAngle (double angle) {
	return Rotation (Quaterniond (
				cos(angle/2), 0, 0, sin(angle/2)));
}

Tensor Rotation::operator *(const Tensor &vector) const {
	Vector3d vector3d;

	// Preprocess 2d vectors
	switch (vector.size(0)) {
	case DIM_2D:
		vector3d[0] = vector[0].item().toDouble ();
		vector3d[1] = vector[1].item().toDouble ();
		vector3d[2] = 0.0;
		break;
	case DIM_3D:
		vector3d = torchVectorToEigen (vector);
		break;
	default:
		assert (false && "Invalid vector size");
		break;
	}

	// Rotate 3d vector
	Vector3d rotated = quaternion * vector3d;

	Tensor torchRotated = vector.clone ();

	torchRotated[0] = rotated[0];
	torchRotated[1] = rotated[1];
	if (vector.size(0) == DIM_3D)
		torchRotated[2] = rotated[2];

	return torchRotated;
}

Rotation Rotation::operator *(const Rotation &other) const {
	return Rotation (quaternion * other.quaternion);
}

ostream &operator<<(ostream &os, const Rotation &dt)
{
	os << dt.quaternion.coeffs ();
	return os;
}
