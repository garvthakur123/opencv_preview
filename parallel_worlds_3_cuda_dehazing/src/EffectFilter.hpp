#ifndef EFFECTFILTER_HPP_
#define EFFECTFILTER_HPP_

#include "SobelFilter.hpp"

/// Implements an effect based on edge detection (SobelColor)
// Darkens color pixels where edge magnitude exceeds a threshold.
class EffectFilter : public ImageFilter {
protected:
	/// device buffer for the 1-channel edge map (Sobel output)
	unsigned char *dEdge;
	/// color depth of edge image in bytes
	unsigned int depthEdge;
public:
	EffectFilter(unsigned int dIn, unsigned int dEdge, unsigned int dOut) :
		ImageFilter(dIn, dOut), dEdge(nullptr), depthEdge(dEdge) {};

	virtual ~EffectFilter() {
		SAFE_CALL(cudaFree(dEdge));
	};

	virtual void resizeBuffers(unsigned int currWidth, unsigned int currHeight) {
		unsigned int bytesEdge = currWidth * currHeight * depthEdge;
		if (currWidth * currHeight > width * height) {
			SAFE_CALL(cudaFree(dEdge));
			SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dEdge), bytesEdge));
		}
		ImageFilter::resizeBuffers(currWidth, currHeight);
	}

	void operator()(const unsigned char *input, const unsigned char *edgeInput,
			unsigned char *output, unsigned int currWidth,
			unsigned int currHeight, float threshold);
};

#endif /* EFFECTFILTER_HPP_ */
