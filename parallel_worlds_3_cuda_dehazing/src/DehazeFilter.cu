/*
 * DehazeFilter.cu  —  optimised Dark Channel Prior (He et al., CVPR 2009)
 *
 * Pipeline (all GPU unless noted):
 *   Pass 1  – separable min-filter on BGR input → dark channel
 *              H-pass: min across BGR channels over horizontal strip  O(2r)
 *              V-pass: min over vertical strip of 1-ch result         O(2r)
 *   CPU     – computeAtmLight: brightest-intensity pixel in top-0.1% dark-ch
 *   Pass 2  – separable min-filter on I/A → raw transmission map      O(2r)
 *              normalise → H-min → V-min → apply ω
 *   Pass 3  – guided filter: refine transmission along edges           O(r_g²)
 *              greyscale guide, per-pixel (a,b), separable box-filter mean_a/b
 *   Pass 4  – scene radiance recovery  J=(I-A)/max(t,tMin)+A          O(1)
 */

#include "DehazeFilter.hpp"

// =========================================================================
// Pass 1 — dark channel, separable min-filter
// =========================================================================

// H-pass: for each pixel, min BGR value over horizontal strip [x-r, x+r]
__global__ void hMinKernel(const unsigned char *bgr, unsigned char *out,
                            unsigned int w, unsigned int h, unsigned int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    unsigned char minVal = 255;
    for (int dx = -(int)r; dx <= (int)r; dx++) {
        int nx = min(max((int)x + dx, 0), (int)w - 1);
        unsigned int idx = (y * w + (unsigned int)nx) * 3;
        unsigned char b = bgr[idx], g = bgr[idx+1], rv = bgr[idx+2];
        unsigned char m = (b < g) ? ((b < rv) ? b : rv) : ((g < rv) ? g : rv);
        if (m < minVal) minVal = m;
    }
    out[y * w + x] = minVal;
}

// V-pass: min over vertical strip [y-r, y+r] of 1-ch H-pass result
__global__ void vMinKernel(const unsigned char *in, unsigned char *out,
                            unsigned int w, unsigned int h, unsigned int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    unsigned char minVal = 255;
    for (int dy = -(int)r; dy <= (int)r; dy++) {
        int ny = min(max((int)y + dy, 0), (int)h - 1);
        unsigned char v = in[(unsigned int)ny * w + x];
        if (v < minVal) minVal = v;
    }
    out[y * w + x] = minVal;
}

// =========================================================================
// Pass 2 — transmission map, separable min-filter on I/A
// =========================================================================

// 2a: normalise BGR by atmospheric light per channel → 3-ch float
__global__ void normalizeByAtmKernel(const unsigned char *bgr, float *normOut,
                                      unsigned int w, unsigned int h,
                                      float atmB, float atmG, float atmR)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    unsigned int pidx = (y * w + x) * 3;
    normOut[pidx]   = (float)bgr[pidx]   / (atmB * 255.0f);
    normOut[pidx+1] = (float)bgr[pidx+1] / (atmG * 255.0f);
    normOut[pidx+2] = (float)bgr[pidx+2] / (atmR * 255.0f);
}

// 2b: H-pass min of 3-ch float over horizontal strip → 1-ch float
__global__ void hMinNormKernel(const float *normBGR, float *out,
                                unsigned int w, unsigned int h, unsigned int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float minVal = 1e10f;
    for (int dx = -(int)r; dx <= (int)r; dx++) {
        int nx = min(max((int)x + dx, 0), (int)w - 1);
        unsigned int pidx = (y * w + (unsigned int)nx) * 3;
        float m = fminf(normBGR[pidx], fminf(normBGR[pidx+1], normBGR[pidx+2]));
        if (m < minVal) minVal = m;
    }
    out[y * w + x] = minVal;
}

// 2c: V-pass min of 1-ch float over vertical strip
__global__ void vMinFloat1Kernel(const float *in, float *out,
                                  unsigned int w, unsigned int h, unsigned int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float minVal = 1e10f;
    for (int dy = -(int)r; dy <= (int)r; dy++) {
        int ny = min(max((int)y + dy, 0), (int)h - 1);
        float v = in[(unsigned int)ny * w + x];
        if (v < minVal) minVal = v;
    }
    out[y * w + x] = minVal;
}

// 2d: t(x) = 1 − ω · dark_norm(x), clamped to [0,1] → uchar
__global__ void applyOmegaKernel(const float *darkNorm, unsigned char *transOut,
                                  unsigned int n, float omega)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = fmaxf(fminf(1.0f - omega * darkNorm[i], 1.0f), 0.0f);
    transOut[i] = (unsigned char)(t * 255.0f);
}

// =========================================================================
// Pass 3 — guided filter (He et al. §4): refine transmission map
//
// Linear model per window k centred at pixel k:
//   a_k = cov(I_k, p_k) / (var(I_k) + ε)
//   b_k = mean(p_k) − a_k · mean(I_k)
// Output pixel i:
//   q_i = mean_{k∋i}(a_k) · I_i + mean_{k∋i}(b_k)
//       = mean_a_i · I_i + mean_b_i
// =========================================================================

// 3a: BGR uchar → 1-ch float greyscale in [0,1]
__global__ void toGreyKernel(const unsigned char *bgr, float *grey,
                              unsigned int w, unsigned int h)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    unsigned int idx = (y * w + x) * 3;
    grey[y * w + x] = (0.114f * bgr[idx] + 0.587f * bgr[idx+1] + 0.299f * bgr[idx+2])
                      / 255.0f;
}

// 3b: uchar [0,255] → float [0,1]  (1-D launch)
__global__ void transToFloatKernel(const unsigned char *in, float *out,
                                    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / 255.0f;
}

// 3c: compute per-pixel guided-filter coefficients (a, b) for a (2r+1)^2 window
//   I : greyscale guide [0,1]
//   p : raw transmission [0,1]
__global__ void guidedComputeABKernel(const float *I, const float *p,
                                       float *a, float *b,
                                       unsigned int w, unsigned int h,
                                       int r, float eps)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sumI = 0, sumP = 0, sumIP = 0, sumII = 0, cnt = 0;
    for (int dy = -r; dy <= r; dy++) {
        int ny = min(max((int)y + dy, 0), (int)h - 1);
        for (int dx = -r; dx <= r; dx++) {
            int nx = min(max((int)x + dx, 0), (int)w - 1);
            float Iv = I[(unsigned int)ny * w + (unsigned int)nx];
            float Pv = p[(unsigned int)ny * w + (unsigned int)nx];
            sumI  += Iv;
            sumP  += Pv;
            sumIP += Iv * Pv;
            sumII += Iv * Iv;
            cnt   += 1.0f;
        }
    }
    float mI  = sumI  / cnt;
    float mP  = sumP  / cnt;
    float ak  = (sumIP / cnt - mI * mP) / (sumII / cnt - mI * mI + eps);
    float bk  = mP - ak * mI;
    a[y * w + x] = ak;
    b[y * w + x] = bk;
}

// 3d/3e: separable box filter, horizontal pass (float)
__global__ void hBoxFloatKernel(const float *in, float *out,
                                  unsigned int w, unsigned int h, int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sum = 0.0f;
    for (int dx = -r; dx <= r; dx++) {
        int nx = min(max((int)x + dx, 0), (int)w - 1);
        sum += in[y * w + (unsigned int)nx];
    }
    out[y * w + x] = sum / (float)(2 * r + 1);
}

// 3d/3e: separable box filter, vertical pass (float)
__global__ void vBoxFloatKernel(const float *in, float *out,
                                  unsigned int w, unsigned int h, int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sum = 0.0f;
    for (int dy = -r; dy <= r; dy++) {
        int ny = min(max((int)y + dy, 0), (int)h - 1);
        sum += in[(unsigned int)ny * w + x];
    }
    out[y * w + x] = sum / (float)(2 * r + 1);
}

// 3f: q = mean_a · I + mean_b  →  refined uchar transmission
__global__ void guidedApplyKernel(const float *meanA, const float *meanB,
                                   const float *I, unsigned char *transOut,
                                   unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float q = fmaxf(fminf(meanA[i] * I[i] + meanB[i], 1.0f), 0.0f);
    transOut[i] = (unsigned char)(q * 255.0f);
}

// =========================================================================
// Pass 4 — recover scene radiance
//   J(x) = (I(x) − A) / max(t(x), tMin) + A
// =========================================================================

__global__ void recoverRadianceKernel(const unsigned char *inImg,
                                       const unsigned char *transIn,
                                       unsigned char *outImg,
                                       unsigned int w, unsigned int h,
                                       float atmB, float atmG, float atmR,
                                       float tMin)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    unsigned int pidx = y * w + x;
    float t = fmaxf((float)transIn[pidx] / 255.0f, tMin);

    unsigned int cidx = pidx * 3;
    float B = ((float)inImg[cidx]   / 255.0f - atmB) / t + atmB;
    float G = ((float)inImg[cidx+1] / 255.0f - atmG) / t + atmG;
    float R = ((float)inImg[cidx+2] / 255.0f - atmR) / t + atmR;

    outImg[cidx]   = (unsigned char)(fmaxf(fminf(B, 1.0f), 0.0f) * 255.0f);
    outImg[cidx+1] = (unsigned char)(fmaxf(fminf(G, 1.0f), 0.0f) * 255.0f);
    outImg[cidx+2] = (unsigned char)(fmaxf(fminf(R, 1.0f), 0.0f) * 255.0f);
}

// =========================================================================
// Host operator — orchestrates the four GPU passes
// =========================================================================

__host__ void DehazeFilter::operator()(const unsigned char *input,
                                        unsigned char *output,
                                        unsigned int w, unsigned int h)
{
    const unsigned int nPix    = w * h;
    const unsigned int bytesIn = nPix * 3;
    const unsigned int blk1D   = (nPix + 255) / 256;
    const int          gr      = (int)guidedRadius;

    prepareBuffers(w, h);

    // Upload BGR frame once — reused by all GPU passes
    SAFE_CALL(cudaMemcpy(dInput, reinterpret_cast<const void*>(input),
                         bytesIn, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Pass 1: dark channel (separable: H then V)
    // ------------------------------------------------------------------
    hMinKernel<<<this->grid, this->threads>>>(dInput, dDarkH, w, h, patchHalf);
    SAFE_CALL(cudaGetLastError());
    vMinKernel<<<this->grid, this->threads>>>(dDarkH, dDark, w, h, patchHalf);
    SAFE_CALL(cudaGetLastError());
    SAFE_CALL(cudaDeviceSynchronize());

    // Download dark channel → CPU for atmospheric light
    std::vector<unsigned char> darkHost(nPix);
    SAFE_CALL(cudaMemcpy(darkHost.data(), dDark, nPix, cudaMemcpyDeviceToHost));
    computeAtmLight(darkHost.data(), input, w, h);

    // ------------------------------------------------------------------
    // Pass 2: transmission (normalise → separable H/V min → apply ω)
    // ------------------------------------------------------------------
    normalizeByAtmKernel<<<this->grid, this->threads>>>(
        dInput, dNormBGR, w, h, atmLight[0], atmLight[1], atmLight[2]);
    SAFE_CALL(cudaGetLastError());

    hMinNormKernel<<<this->grid, this->threads>>>(dNormBGR, dNormH, w, h, patchHalf);
    SAFE_CALL(cudaGetLastError());

    vMinFloat1Kernel<<<this->grid, this->threads>>>(dNormH, dNormDark, w, h, patchHalf);
    SAFE_CALL(cudaGetLastError());

    applyOmegaKernel<<<blk1D, 256>>>(dNormDark, dTrans, nPix, omega);
    SAFE_CALL(cudaGetLastError());
    SAFE_CALL(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Pass 3: guided filter — edge-preserving transmission refinement
    // ------------------------------------------------------------------
    toGreyKernel<<<this->grid, this->threads>>>(dInput, dGrey, w, h);
    SAFE_CALL(cudaGetLastError());

    transToFloatKernel<<<blk1D, 256>>>(dTrans, dTransF, nPix);
    SAFE_CALL(cudaGetLastError());

    // per-pixel (a, b) from guided linear model in r-neighbourhood
    guidedComputeABKernel<<<this->grid, this->threads>>>(
        dGrey, dTransF, dA, dB, w, h, gr, guidedEps);
    SAFE_CALL(cudaGetLastError());

    // mean_a  (separable box filter on a)
    hBoxFloatKernel<<<this->grid, this->threads>>>(dA,   dTmp,   w, h, gr);
    SAFE_CALL(cudaGetLastError());
    vBoxFloatKernel<<<this->grid, this->threads>>>(dTmp, dMeanA, w, h, gr);
    SAFE_CALL(cudaGetLastError());

    // mean_b  (separable box filter on b)
    hBoxFloatKernel<<<this->grid, this->threads>>>(dB,   dTmp,   w, h, gr);
    SAFE_CALL(cudaGetLastError());
    vBoxFloatKernel<<<this->grid, this->threads>>>(dTmp, dMeanB, w, h, gr);
    SAFE_CALL(cudaGetLastError());

    // q = mean_a * I + mean_b → refined transmission (uchar, overwrites dTrans)
    guidedApplyKernel<<<blk1D, 256>>>(dMeanA, dMeanB, dGrey, dTrans, nPix);
    SAFE_CALL(cudaGetLastError());
    SAFE_CALL(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Pass 4: recover scene radiance
    // ------------------------------------------------------------------
    recoverRadianceKernel<<<this->grid, this->threads>>>(
        dInput, dTrans, dOutput, w, h,
        atmLight[0], atmLight[1], atmLight[2], tMin);
    SAFE_CALL(cudaGetLastError());

    SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output), dOutput,
                         bytesIn, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());
}
