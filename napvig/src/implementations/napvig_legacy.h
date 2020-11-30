#ifndef NAPVIG_LEGACY_H
#define NAPVIG_LEGACY_H

#include "../napvig.h"

class NapvigLegacy : public Napvig
{
	boost::optional<Trajectory> trajectoryAlgorithm (const State &zeroState) override;

public:
	NapvigLegacy(const std::shared_ptr<Landscape::Params> &landscapeParams,
				 const std::shared_ptr<Napvig::Params> &napvigParams);


};

#endif // NAPVIG_LEGACY_H
