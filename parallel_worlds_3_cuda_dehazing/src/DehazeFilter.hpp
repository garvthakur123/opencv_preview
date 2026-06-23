#ifndef DEHAZEFILTER_HPP_
#define DEHAZEFILTER_HPP_

#include <algorithm>

#include "ImageFilter.hpp"

/// Implements Dark Channel Prior dehazing (He et al., CVPR 2009) as a
/// fully-GPU four-pass CUDA pipeline:
///   1. darkChannelKernel          (GPU) -> 1-ch dark map
///   2. histogramKernel            (GPU) -> 256-bin histogram of dark map
///      atmLightSumKernel          (GPU) -> per-channel BGR sums of top pixels
///      [CPU: 3 divisions + clamp from 1056-byte download -> atmLight[3]]
///   3. estimateTransmissionKernel (GPU) -> 1-ch transmission map
///   4. recoverRadianceKernel      (GPU) -> BGR dehazed output
///
/// Previously computeAtmLight ran entirely on the CPU, downloading ~300 KB of
/// dark-channel data per frame.  The GPU histogram + sum approach reduces the
/// PCIe transfer to 1056 bytes (1024 histogram + 32 sums) per frame.
class DehazeFilter : public ImageFilter {
protected:
    /// device buffer: 1 byte per pixel (dark channel map from pass 1)
    unsigned char *dDark;
    /// device buffer: 1 byte per pixel (transmission map from pass 3)
    unsigned char *dTrans;

    /// device buffer: 256-bin histogram of the dark channel (fixed size)
    unsigned int *dHist;
    /// device buffer: [sumB, sumG, sumR, count] for atmospheric light (fixed size)
    unsigned long long *dSums;

    /// atmospheric light per BGR channel in [0,1], updated each frame
    float atmLight[3];

    /// half-side of the min-filter patch window (default 7 -> 15x15 patch)
    unsigned int patchHalf;
    /// haze-retention factor: higher = more aggressive dehazing
    float omega;
    /// minimum transmission clamp to prevent divide-by-near-zero
    float tMin;

public:
    /// @param patchHalf_  half-side of the DCP patch window (default 7)
    /// @param omega_      haze retention factor (default 0.95)
    /// @param tMin_       minimum transmission clamp (default 0.1)
    DehazeFilter(unsigned int patchHalf_ = 7,
                 float omega_            = 0.95f,
                 float tMin_             = 0.10f)
        : ImageFilter(3, 3),
          dDark(nullptr), dTrans(nullptr),
          dHist(nullptr), dSums(nullptr),
          patchHalf(patchHalf_), omega(omega_), tMin(tMin_)
    {
        atmLight[0] = atmLight[1] = atmLight[2] = 1.0f;
    }

    virtual ~DehazeFilter() {
        SAFE_CALL(cudaFree(dDark));
        SAFE_CALL(cudaFree(dTrans));
        SAFE_CALL(cudaFree(dHist));
        SAFE_CALL(cudaFree(dSums));
    }

    /// Allocate (or grow) all device buffers for the given image size.
    /// dHist and dSums are fixed-size and only allocated once on first call.
    virtual void resizeBuffers(unsigned int currWidth, unsigned int currHeight) {
        unsigned int nPix = currWidth * currHeight;
        if (nPix > width * height) {
            SAFE_CALL(cudaFree(dDark));
            SAFE_CALL(cudaFree(dTrans));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dDark),  nPix));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dTrans), nPix));

            // dHist and dSums are fixed size regardless of image dimensions;
            // free + realloc only on the first call (width==0 && height==0)
            SAFE_CALL(cudaFree(dHist));
            SAFE_CALL(cudaFree(dSums));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dHist),
                                 256 * sizeof(unsigned int)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dSums),
                                 4 * sizeof(unsigned long long)));
        }
        // allocates dInput (3ch BGR) and dOutput (3ch BGR), updates width/height
        ImageFilter::resizeBuffers(currWidth, currHeight);
    }

    void operator()(const unsigned char *input, unsigned char *output,
                    unsigned int w, unsigned int h);
};

#endif /* DEHAZEFILTER_HPP_ */
