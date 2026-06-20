#ifndef DEHAZEFILTER_HPP_
#define DEHAZEFILTER_HPP_

#include <vector>
#include <algorithm>

#include "ImageFilter.hpp"

/// Implements Dark Channel Prior dehazing (He et al., CVPR 2009) as a
/// three-pass CUDA pipeline:
///   1. darkChannelKernel       (GPU)  -> 1-ch dark map
///   2. computeAtmLight                (CPU)  -> atmospheric light A[3]
///   3. estimateTransmissionKernel (GPU)  -> 1-ch transmission map
///   4. recoverRadianceKernel      (GPU)  -> BGR dehazed output
///
/// Follows the same class pattern as EffectFilter: subclass of ImageFilter
/// with additional device buffers and a CPU helper step in the middle.
class DehazeFilter : public ImageFilter {
protected:
    /// device buffer: 1 byte per pixel (dark channel map from pass 1)
    unsigned char *dDark;
    /// device buffer: 1 byte per pixel (transmission map from pass 2)
    unsigned char *dTrans;

    /// atmospheric light per BGR channel in [0,1], updated each frame
    float atmLight[3];

    /// half-side of the min-filter patch window (default 7 -> 15x15 patch)
    unsigned int patchHalf;
    /// haze-retention factor: lower = less aggressive dehazing (default 0.75)
    float omega;
    /// minimum transmission clamp to prevent divide-by-near-zero (default 0.2)
    float tMin;

    /// Estimate atmospheric light on the CPU from the dark channel and the
    /// original colour frame.  Top 0.1% of dark-channel pixels are candidates;
    /// A is the per-channel mean of all candidates, capped to prevent a single
    /// bright highlight from causing extreme colour shifts on non-hazy scenes.
    void computeAtmLight(const unsigned char *dark,
                         const unsigned char *color,
                         unsigned int w, unsigned int h)
    {
        unsigned int n    = w * h;
        unsigned int nTop = std::max(1u, n / 1000);

        // find the threshold via partial sort on a copy
        std::vector<unsigned char> vals(dark, dark + n);
        std::nth_element(vals.begin(), vals.begin() + (n - nTop), vals.end());
        unsigned char thresh = vals[n - nTop];

        // accumulate per-channel mean across all qualifying candidates;
        // using mean rather than a single brightest pixel is far more robust
        // when coloured highlights (windows, lamps) are present
        unsigned long sumB = 0, sumG = 0, sumR = 0, count = 0;
        for (unsigned int i = 0; i < n; i++) {
            if (dark[i] >= thresh) {
                sumB += color[i * 3];
                sumG += color[i * 3 + 1];
                sumR += color[i * 3 + 2];
                count++;
            }
        }
        if (count == 0) count = 1;

        // floor at 1/255 (avoid divide-by-zero in kernel),
        // ceiling at 0.85 (avoid extreme subtraction on non-hazy scenes)
        const float capVal = 0.85f;
        atmLight[0] = std::min(std::max(sumB / (count * 255.0f), 1.0f / 255.0f), capVal);
        atmLight[1] = std::min(std::max(sumG / (count * 255.0f), 1.0f / 255.0f), capVal);
        atmLight[2] = std::min(std::max(sumR / (count * 255.0f), 1.0f / 255.0f), capVal);
    }

public:
    /// @param patchHalf_  half-side of the DCP patch window (default 7)
    /// @param omega_      haze retention factor (default 0.75)
    /// @param tMin_       minimum transmission clamp (default 0.2)
    DehazeFilter(unsigned int patchHalf_ = 7,
                 float omega_ = 0.75f,
                 float tMin_  = 0.2f)
        : ImageFilter(3, 3),
          dDark(nullptr), dTrans(nullptr),
          patchHalf(patchHalf_), omega(omega_), tMin(tMin_)
    {
        atmLight[0] = atmLight[1] = atmLight[2] = 1.0f;
    }

    virtual ~DehazeFilter() {
        SAFE_CALL(cudaFree(dDark));
        SAFE_CALL(cudaFree(dTrans));
    }

    /// Allocate (or grow) all device buffers for the given image size.
    virtual void resizeBuffers(unsigned int currWidth, unsigned int currHeight) {
        unsigned int nPix = currWidth * currHeight;
        if (nPix > width * height) {
            SAFE_CALL(cudaFree(dDark));
            SAFE_CALL(cudaFree(dTrans));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dDark),  nPix));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dTrans), nPix));
        }
        // allocates dInput (3ch BGR) and dOutput (3ch BGR), updates width/height
        ImageFilter::resizeBuffers(currWidth, currHeight);
    }

    void operator()(const unsigned char *input, unsigned char *output,
                    unsigned int w, unsigned int h);
};

#endif /* DEHAZEFILTER_HPP_ */
