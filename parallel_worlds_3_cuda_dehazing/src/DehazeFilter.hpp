#ifndef DEHAZEFILTER_HPP_
#define DEHAZEFILTER_HPP_

#include <vector>
#include <algorithm>

#include "ImageFilter.hpp"

/// Dark Channel Prior dehazing — He et al., CVPR 2009 — four GPU passes:
///   1. Separable min-filter (BGR → dark channel)          O(2r) per pixel
///   2. Separable min-filter on I/A (→ raw transmission)   O(2r) per pixel
///   3. Guided filter (edge-preserving transmission refine) O(r_g^2) per pixel
///   4. Radiance recovery  J = (I − A) / max(t, tMin) + A  O(1) per pixel
///
/// Quality improvements over the original version:
///   • Separable min-filters replace O(r^2) 2D patch → O(2r) per pixel each
///   • Atmospheric light = single brightest pixel inside top-0.1% dark-channel
///     candidates (He et al.) instead of the mean — more robust to highlights
///   • Guided filter refines the blocky raw transmission using the greyscale
///     input as a structure guide, eliminating halo artefacts at depth edges
///   • omega raised to 0.95 (was 0.75) for more aggressive dehazing
///   • tMin lowered to 0.1 (was 0.2) to allow correction in dense haze
class DehazeFilter : public ImageFilter {
protected:
    // --- dark-channel buffers ---
    unsigned char *dDark;       // 1ch uchar  dark channel (pass 1 output)
    unsigned char *dDarkH;      // 1ch uchar  H-pass intermediate

    // --- transmission buffers ---
    float         *dNormBGR;    // 3ch float  I/A  (normalised by atm light)
    float         *dNormH;      // 1ch float  H-pass min of dNormBGR
    float         *dNormDark;   // 1ch float  dark channel of normalised image
    unsigned char *dTrans;      // 1ch uchar  refined transmission [0..255]

    // --- guided-filter buffers ---
    float *dGrey;               // 1ch float  greyscale guide [0,1]
    float *dTransF;             // 1ch float  raw transmission [0,1]
    float *dA, *dB;             // 1ch float  per-pixel window coefficients
    float *dMeanA, *dMeanB;     // 1ch float  box-filtered coefficients
    float *dTmp;                // 1ch float  H-pass temp for box filter

    float        atmLight[3];   // BGR atmospheric light in [0,1]
    unsigned int patchHalf;     // half-side of DCP patch window
    unsigned int guidedRadius;  // guided-filter neighbourhood radius
    float        omega;         // haze-retention factor
    float        tMin;          // lower clamp on transmission
    float        guidedEps;     // guided-filter regularisation epsilon

    /// Atmospheric light: He et al. pick the pixel with the highest intensity
    /// in the original image among the top 0.1% brightest dark-channel pixels.
    /// This is more robust than a mean because a single bright lamp or sky
    /// patch cannot dominate the estimate.
    void computeAtmLight(const unsigned char *dark,
                         const unsigned char *color,
                         unsigned int w, unsigned int h)
    {
        unsigned int n    = w * h;
        unsigned int nTop = std::max(1u, n / 1000);

        std::vector<unsigned char> vals(dark, dark + n);
        std::nth_element(vals.begin(), vals.begin() + (n - nTop), vals.end());
        unsigned char thresh = vals[n - nTop];

        float maxI = -1.0f;
        unsigned int best = 0;
        for (unsigned int i = 0; i < n; i++) {
            if (dark[i] >= thresh) {
                float I = 0.299f * color[i*3+2]
                        + 0.587f * color[i*3+1]
                        + 0.114f * color[i*3];
                if (I > maxI) { maxI = I; best = i; }
            }
        }

        const float capVal = 0.95f;
        for (int c = 0; c < 3; c++) {
            float v = color[best*3 + c] / 255.0f;
            atmLight[c] = std::max(std::min(v, capVal), 1.0f / 255.0f);
        }
    }

public:
    DehazeFilter(unsigned int patchHalf_    = 7,
                 float        omega_        = 0.95f,
                 float        tMin_         = 0.1f,
                 unsigned int guidedRadius_ = 8,
                 float        guidedEps_    = 0.001f)
        : ImageFilter(3, 3),
          dDark(nullptr), dDarkH(nullptr),
          dNormBGR(nullptr), dNormH(nullptr), dNormDark(nullptr),
          dTrans(nullptr),
          dGrey(nullptr), dTransF(nullptr),
          dA(nullptr), dB(nullptr),
          dMeanA(nullptr), dMeanB(nullptr), dTmp(nullptr),
          patchHalf(patchHalf_), guidedRadius(guidedRadius_),
          omega(omega_), tMin(tMin_), guidedEps(guidedEps_)
    {
        atmLight[0] = atmLight[1] = atmLight[2] = 1.0f;
    }

    virtual ~DehazeFilter() {
        SAFE_CALL(cudaFree(dDark));
        SAFE_CALL(cudaFree(dDarkH));
        SAFE_CALL(cudaFree(dNormBGR));
        SAFE_CALL(cudaFree(dNormH));
        SAFE_CALL(cudaFree(dNormDark));
        SAFE_CALL(cudaFree(dTrans));
        SAFE_CALL(cudaFree(dGrey));
        SAFE_CALL(cudaFree(dTransF));
        SAFE_CALL(cudaFree(dA));
        SAFE_CALL(cudaFree(dB));
        SAFE_CALL(cudaFree(dMeanA));
        SAFE_CALL(cudaFree(dMeanB));
        SAFE_CALL(cudaFree(dTmp));
    }

    virtual void resizeBuffers(unsigned int currWidth, unsigned int currHeight) {
        unsigned int nPix = currWidth * currHeight;
        if (nPix > width * height) {
            SAFE_CALL(cudaFree(dDark));
            SAFE_CALL(cudaFree(dDarkH));
            SAFE_CALL(cudaFree(dNormBGR));
            SAFE_CALL(cudaFree(dNormH));
            SAFE_CALL(cudaFree(dNormDark));
            SAFE_CALL(cudaFree(dTrans));
            SAFE_CALL(cudaFree(dGrey));
            SAFE_CALL(cudaFree(dTransF));
            SAFE_CALL(cudaFree(dA));
            SAFE_CALL(cudaFree(dB));
            SAFE_CALL(cudaFree(dMeanA));
            SAFE_CALL(cudaFree(dMeanB));
            SAFE_CALL(cudaFree(dTmp));

            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dDark),     nPix));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dDarkH),    nPix));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dNormBGR),  nPix * 3 * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dNormH),    nPix * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dNormDark), nPix * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dTrans),    nPix));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dGrey),     nPix * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dTransF),   nPix * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dA),        nPix * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dB),        nPix * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dMeanA),    nPix * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dMeanB),    nPix * sizeof(float)));
            SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dTmp),      nPix * sizeof(float)));
        }
        ImageFilter::resizeBuffers(currWidth, currHeight);
    }

    void operator()(const unsigned char *input, unsigned char *output,
                    unsigned int w, unsigned int h);
};

#endif /* DEHAZEFILTER_HPP_ */
