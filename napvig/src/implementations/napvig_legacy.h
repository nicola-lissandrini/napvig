#ifndef NAPVIG_LEGACY_H
#define NAPVIG_LEGACY_H

#include "../napvig.h"

class NapvigLegacy : public Napvig
{
	boost::optional<Trajectory> trajectoryAlgorithm (const State &zeroState);

public:
	NapvigLegacy(Landscape::Params &landscapeParams,
				 Napvig::Params &napvigParams);


};

#endif // NAPVIG_LEGACY_H
