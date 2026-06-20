#ifndef DEHAZEFILTER_HPP_
#define DEHAZEFILTER_HPP_

#include <vector>
#include <algorithm>
#include <cstring>

#include "ImageFilter.hpp"

/// Implements Dark Channel Prior dehazing (He et al., CVPR 2009) as a
/// three-pass CUDA pipeline:
///   1. darkChannelKernel          (GPU)  -> 1-ch dark map
///   2. computeAtmLight            (CPU)  -> atmospheric light A[3]
///   3. estimateTransmissionKernel (GPU)  -> 1-ch transmission map
///   4. recoverRadianceKernel      (GPU)  -> BGR dehazed output
///
/// Improvements over the original OpenCL version:
///   - Input brightness normalisation: stretches dark images to a useful
///     dynamic range before DCP runs, so the algorithm finds actual haze.
///   - Luminance-neutral atmLight blending: reduces per-channel colour casts
///     caused by strongly coloured atmospheric light (sunsets, artificial light).
///   - Output gamma correction: additional brightening of the dehazed result.
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
    /// haze-retention factor: lower = less aggressive dehazing
    float omega;
    /// minimum transmission clamp to prevent divide-by-near-zero
    float tMin;
    /// output gamma (> 1 brightens; applied as output^(1/gamma))
    float gamma;

    /// Reusable CPU buffer for the brightness-normalised input copy.
    /// Avoids a heap allocation on every frame.
    std::vector<unsigned char> normBuf;

    // -----------------------------------------------------------------------
    // normalizeInput
    // -----------------------------------------------------------------------
    /// Scale the input image so its 99th-percentile brightness reaches
    /// ~220, making the dark channel meaningful for underexposed images.
    /// For images that are already bright (p99 >= 200) no scaling is applied.
    /// The result is written into `normBuf`; the caller should use
    /// normBuf.data() as the input to all subsequent steps.
    void normalizeInput(const unsigned char *input, unsigned int nPix)
    {
        // fast O(N) histogram-based 99th-percentile -- no extra allocation
        unsigned int hist[256] = {};
        for (unsigned int i = 0; i < nPix * 3; i++)
            hist[input[i]]++;

        const unsigned int target = std::max(1u, (unsigned int)(nPix * 3 * 0.01f));
        unsigned int count = 0;
        unsigned char p99  = 255;
        for (int v = 255; v >= 0; v--) {
            count += hist[v];
            if (count >= target) { p99 = (unsigned char)v; break; }
        }

        // only stretch when image is noticeably darker than target brightness;
        // cap scale at 4× to avoid amplifying sensor noise in very dark shots
        const unsigned char targetBright = 180;
        float scale = (p99 > 10 && p99 < targetBright)
                      ? std::min(static_cast<float>(targetBright) / p99, 4.0f)
                      : 1.0f;

        normBuf.resize(nPix * 3);
        if (scale == 1.0f) {
            std::memcpy(normBuf.data(), input, nPix * 3);
        } else {
            for (unsigned int i = 0; i < nPix * 3; i++) {
                float v = input[i] * scale;
                normBuf[i] = (v >= 255.0f) ? 255u : static_cast<unsigned char>(v);
            }
        }
    }

    // -----------------------------------------------------------------------
    // computeAtmLight
    // -----------------------------------------------------------------------
    /// Estimate atmospheric light from the dark channel and the (normalised)
    /// colour frame.  Top 0.1% of dark-channel pixels are candidates; A is
    /// the per-channel mean, then partially blended toward the luminance-
    /// neutral value to suppress colour casts from coloured light sources.
    void computeAtmLight(const unsigned char *dark,
                         const unsigned char *color,
                         unsigned int w, unsigned int h)
    {
        unsigned int n    = w * h;
        unsigned int nTop = std::max(1u, n / 1000);

        // find threshold via partial sort on a copy
        std::vector<unsigned char> vals(dark, dark + n);
        std::nth_element(vals.begin(), vals.begin() + (n - nTop), vals.end());
        unsigned char thresh = vals[n - nTop];

        // per-channel mean of all candidate pixels
        unsigned long sumB = 0, sumG = 0, sumR = 0, cnt = 0;
        for (unsigned int i = 0; i < n; i++) {
            if (dark[i] >= thresh) {
                sumB += color[i * 3];
                sumG += color[i * 3 + 1];
                sumR += color[i * 3 + 2];
                cnt++;
            }
        }
        if (cnt == 0) cnt = 1;

        // floor at 1/255 (prevent division by zero in kernel)
        // ceiling at 0.92 (prevent extreme subtraction artefacts)
        const float capVal = 0.92f;
        atmLight[0] = std::min(std::max(sumB / (cnt * 255.0f), 1.0f / 255.0f), capVal);
        atmLight[1] = std::min(std::max(sumG / (cnt * 255.0f), 1.0f / 255.0f), capVal);
        atmLight[2] = std::min(std::max(sumR / (cnt * 255.0f), 1.0f / 255.0f), capVal);

        // Blend 30% toward the luminance-neutral value.
        // This damps colour casts that arise when the atmospheric light is
        // strongly coloured (sunset orange, artificial yellow, etc.) -- the
        // per-channel formula otherwise overcorrects whichever channel is
        // furthest from the mean.
        float lum = 0.0722f * atmLight[0]
                  + 0.7152f * atmLight[1]
                  + 0.2126f * atmLight[2];
        const float blend = 0.30f;
        atmLight[0] = (1.0f - blend) * atmLight[0] + blend * lum;
        atmLight[1] = (1.0f - blend) * atmLight[1] + blend * lum;
        atmLight[2] = (1.0f - blend) * atmLight[2] + blend * lum;
    }

public:
    /// @param patchHalf_  half-side of the DCP patch window (default 7 -> 15x15)
    /// @param omega_      haze retention factor; higher = more aggressive (default 0.90)
    /// @param tMin_       minimum transmission clamp (default 0.10)
    /// @param gamma_      output gamma boost; 1.0 = off, 1.3 = mild brightening (default 1.3)
    DehazeFilter(unsigned int patchHalf_ = 7,
                 float omega_            = 0.90f,
                 float tMin_             = 0.10f,
                 float gamma_            = 1.30f)
        : ImageFilter(3, 3),
          dDark(nullptr), dTrans(nullptr),
          patchHalf(patchHalf_), omega(omega_), tMin(tMin_), gamma(gamma_)
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
