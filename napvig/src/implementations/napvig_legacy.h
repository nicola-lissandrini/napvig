#ifndef NAPVIG_LEGACY_H
#define NAPVIG_LEGACY_H

#include "../napvig.h"

class NapvigLegacy : public Napvig
{
	boost::optional<Trajectory> trajectoryAlgorithm (const State &zeroState) override;

public:
	NapvigLegacy(const Landscape::Params::Ptr &landscapeParams,
				 const Params::Ptr &napvigParams);

	DEF_SHARED(NapvigLegacy)
};

#endif // NAPVIG_LEGACY_H
