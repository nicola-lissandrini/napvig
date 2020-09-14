#include "napvig_map.h"

#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <torch/autograd.h>
#include <ATen/Layout.h>

using namespace torch;
using namespace torch::indexing;
using namespace std;

#define ARGMIN

double NapvigMap::getNoAmplificationGain() const {
	const double sqMR = pow(params.measureRadius,2);
	const double sqSR = pow(params.smoothRadius,2);
	return pow ((sqMR + sqSR)/
			(2 * M_PI * sqMR * sqSR), 2/double(params.dim));
}

double NapvigMap::getSmoothGain () const {
	return pow (2 * M_PI * pow(params.smoothRadius,2), double(params.dim)/2) * getNoAmplificationGain ();
}


void dump (autograd::profiler::thread_event_lists &events) {
	cout << "Profile dump:" << endl;
	for (auto &currI : events) {
		cout << "Thread" << endl;
		auto old = currI[0];
		auto first = old;
		string name = currI[0].name ();
		for (auto &currJ : currI) {
			if (currJ.kind () == "pop") {
				cout << name << ": " << old.cpu_elapsed_us (currJ) << "us" << endl;
				old = currJ;
			} else
				name = currJ.name ();
		}
		cout << "Total: " << first.cpu_elapsed_us (old) << "us" << endl;
	}
	cout << endl << endl;
}

#ifdef GAMMA
Tensor NapvigMap::gaussian(const Tensor &x, double sigma) const {
	Tensor x1 = x.index({Ellipsis, 0, Ellipsis});
	Tensor x2 = x.index({Ellipsis, 1, Ellipsis});
	Tensor nrm = x1.pow(2) + x2.pow(2);
	Tensor ret = (nrm*(1/(-2*sigma*sigma))).exp();

	return ret;
}
#elif defined ARGMIN
Tensor NapvigMap::gaussian(const Tensor &x, double sigma) const {
	return (x * (-0.5/(sigma*sigma))). exp();
}
#endif

Tensor NapvigMap::gamma(const Tensor &x) const {
	return gaussian (x - measures, params.measureRadius);
}

Tensor NapvigMap::normSquare (const Tensor &x) const {
	return x.pow(2).sum(2);
}

#ifdef GAMMA
Tensor NapvigMap::potentialFunction(const Tensor &x) const {
	Tensor val, boh;
	tie (val, std::ignore) = gamma(x).max (0);

	return val;
}
#elif defined ARGMIN
Tensor NapvigMap::potentialFunction(const Tensor &x) const {
	Tensor index;
	Tensor distToMeasures = normSquare(x - measures);
	index = distToMeasures.argmin (0)[0];

	return exponentialPart (distToMeasures, index);
}

Tensor NapvigMap::exponentialPart (const Tensor &distToMeasures, const Tensor &kIndex) const {
	Tensor collapsed = distToMeasures.index ({kIndex, Ellipsis});

	return gaussian (collapsed, params.measureRadius);
}

Tensor NapvigMap::exponentialPart (const Tensor &selectedDistToMeasures) const {
	Tensor collapsed = selectedDistToMeasures;

	return gaussian (collapsed, params.measureRadius);
}
#endif


Tensor NapvigMap::preSmooth(const Tensor &x) const {
	return potentialFunction (x) * smoothGain;
}

NapvigMap::NapvigMap(const NapvigMap::Params &_params):
	params(_params)
{
	smoothGain = getSmoothGain ();
	flags.addFlag ("measures_set");
}

Tensor NapvigMap::value (const Tensor &x) const {
	if (!flags.isReady ())
		return Tensor ();

	if (measures.size(0) == 0)
		return torch::zeros ({1}, kDouble)[0];
	else
		return montecarlo (&NapvigMap::preSmooth, this, params.dim, params.precision, x, params.smoothRadius);
}

#ifdef AUTOGRAD
Tensor NapvigMap::grad (const Tensor &x) const
{
	Tensor v;

	x.requires_grad_ (true);
	v = value (x);
	v.backward ();

	return x.grad ().detach ();
}
#else
Tensor NapvigMap::preSmoothGrad (const Tensor &x) const
{
	Tensor measuresDiff = (x - measures);
	Tensor distToMeasures = normSquare (measuresDiff);
	Tensor collapsedDist, idxes;
	tie (collapsedDist, idxes) = distToMeasures.min (0);

	Tensor collapsedDiff = measuresDiff.permute ({1,0,2}).index ({at::arange(measuresDiff.size (1)), idxes, Ellipsis});

	Tensor ret = collapsedDiff * exponentialPart (collapsedDist).unsqueeze (1) * smoothGain;

	return ret;
}

Tensor NapvigMap::grad (const Tensor &x) const {
	if (measures.size(0) == 0)
		return torch::zeros ({2}, kDouble);
	else
		return montecarlo (&NapvigMap::preSmoothGrad, this, params.dim, params.precision, x.view({1,params.dim}), params.smoothRadius);
}

double NapvigMap::gammaDistance (double distance) const {
		return exp (-pow(distance,2)/(2 * pow(params.measureRadius,2)));
}

#endif

bool NapvigMap::isReady() const {
	return flags.isReady ();
}

int NapvigMap::getDim() const {
	return params.dim;
}

#if 0
void NapvigMap::setMeasures (const Tensor &newMeasures)
{
	const int measuresCount = newMeasures.size(0);
	const int measuresDim = newMeasures.size (1);
	const int clustersMax = measuresCount;
	Tensor clusters = torch::empty ({clustersMax, measuresDim});
	Tensor lastClusterStart = torch::empty ({measuresDim});
	int currClusterNo = 0;
	int lastClusterSize = 0;

	lastClusterStart = newMeasures.index({0, Ellipsis});
	clusters.index_put_ ({0, 0, Ellipsis}, lastClusterStart);

	for (int i = 0; i < newMeasures.size(0); i++) {
		Tensor curr = newMeasures.index ({i, Ellipsis});
		cout << torch::isfinite (curr) << endl;
		if ((curr - lastClusterStart).norm ()[0].item().toDouble () < params.measureRadius) {
			if (lastClusterSize > 0) {
				currClusterNo++;
				lastClusterSize = 1;
				clusters.index_put_ ({currClusterNo, Ellipsis}, curr);
			} else {
				Tensor lastCluster = clusters.index ({currClusterNo, Ellipsis});
				clusters.index_put_ ({currClusterNo, Ellipsis}, lastCluster + (curr - lastCluster)/double(lastClusterSize));
			}
		} else
			lastClusterSize = 0;
	}

	//Tensor validIdxes = (torch::isfinite (newMeasures).sum(1)).nonzero ();

	//measures = newMeasures.index ({validIdxes}).reshape ({validIdxes.size (0), 1, dim});
	//measures = measures.index_select (0,{torch::arange (0, measures.size(0),10, torch::kInt64)});

	measures = clusters.view ({currClusterNo, measuresDim});
	flags.set ("measures_set");
}
#else

void NapvigMap::setMeasures (const Tensor &newMeasures)
{
	Tensor validIdxes = (torch::isfinite (newMeasures).sum(1)).nonzero ();

	measures = newMeasures.index ({validIdxes}).view ({validIdxes.size (0), 1, params.dim});
	measures = measures.index ({Slice(None,None,3), Ellipsis});

	flags.set ("measures_set");
}
#endif
