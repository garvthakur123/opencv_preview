/*
 * DehazeFilter.cu
 *
 * CUDA port of dehazeFilters.cl (opencv_preview1/parallel_worlds_3).
 * Dark Channel Prior dehazing — He et al., CVPR 2009.
 *
 * Three GPU passes:
 *   1. darkChannelKernel          : min BGR value over local patch -> dark map
 *   2. estimateTransmissionKernel : normalize by atm light, run dark channel -> t map
 *   3. recoverRadianceKernel      : J = (I - A) / max(t, tMin) + A -> dehazed BGR
 *
 * Between pass 1 and 2 the dark map is downloaded to the CPU so that
 * computeAtmLight() (defined in DehazeFilter.hpp) can derive the atmospheric
 * light estimate.
 */

#include "DehazeFilter.hpp"

// ----- Pass 1 ---------------------------------------------------------------
// Compute the dark channel: for each pixel, find the minimum value across all
// three BGR channels within a (2*patchHalf+1)^2 neighbourhood.
// @param inImg    BGR input, 3 bytes per pixel
// @param darkOut  1-byte-per-pixel dark channel output
// @param w/h      image dimensions
// @param patchHalf  half-size of the patch window (e.g. 7 -> 15x15 window)
__global__ void darkChannelKernel(const unsigned char *inImg,
                                  unsigned char *darkOut,
                                  unsigned int w, unsigned int h,
                                  unsigned int patchHalf)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    unsigned char minVal = 255;
    int ph = (int)patchHalf;

    for (int dy = -ph; dy <= ph; dy++) {
        for (int dx = -ph; dx <= ph; dx++) {
            int nx = min(max((int)x + dx, 0), (int)w - 1);
            int ny = min(max((int)y + dy, 0), (int)h - 1);
            unsigned int idx = ((unsigned int)ny * w + (unsigned int)nx) * 3;
            unsigned char b = inImg[idx];
            unsigned char g = inImg[idx + 1];
            unsigned char r = inImg[idx + 2];
            unsigned char minCh = (b < g) ? ((b < r) ? b : r) : ((g < r) ? g : r);
            if (minCh < minVal) minVal = minCh;
        }
    }
    darkOut[y * w + x] = minVal;
}

// ----- Pass 2 ---------------------------------------------------------------
// Estimate transmission map: normalise each neighbour pixel by atmospheric
// light, compute the dark channel of the normalised image, then apply
//   t(x) = 1 - omega * darkChannel(I/A)(x)
// Result stored as unsigned char (0 = 0.0, 255 = 1.0).
// @param inImg    BGR input, 3 bytes per pixel
// @param transOut  1-byte-per-pixel transmission map
// @param w/h      image dimensions
// @param atmB/atmG/atmR  atmospheric light per channel in [0,1]
// @param patchHalf  same window size as darkChannelKernel
// @param omega    haze retention factor (typically 0.75)
__global__ void estimateTransmissionKernel(const unsigned char *inImg,
                                           unsigned char *transOut,
                                           unsigned int w, unsigned int h,
                                           float atmB, float atmG, float atmR,
                                           unsigned int patchHalf,
                                           float omega)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float minVal = 1.0f;
    int ph = (int)patchHalf;

    for (int dy = -ph; dy <= ph; dy++) {
        for (int dx = -ph; dx <= ph; dx++) {
            int nx = min(max((int)x + dx, 0), (int)w - 1);
            int ny = min(max((int)y + dy, 0), (int)h - 1);
            unsigned int idx = ((unsigned int)ny * w + (unsigned int)nx) * 3;
            float normB = (float)inImg[idx]     / (atmB * 255.0f);
            float normG = (float)inImg[idx + 1] / (atmG * 255.0f);
            float normR = (float)inImg[idx + 2] / (atmR * 255.0f);
            float m = fminf(normB, fminf(normG, normR));
            if (m < minVal) minVal = m;
        }
    }
    float t = 1.0f - omega * minVal;
    transOut[y * w + x] = (unsigned char)(fmaxf(fminf(t, 1.0f), 0.0f) * 255.0f);
}

// ----- Pass 3 ---------------------------------------------------------------
// Recover scene radiance:
//   J(x) = (I(x) - A) / max(t(x), tMin) + A
// @param inImg   BGR input, 3 bytes per pixel
// @param transIn  1-byte transmission map from estimateTransmissionKernel
// @param outImg  BGR output, 3 bytes per pixel
// @param w/h     image dimensions
// @param atmB/atmG/atmR  atmospheric light per channel in [0,1]
// @param tMin    lower clamp on transmission to avoid division by near-zero
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
    float B = ((float)inImg[cidx]     / 255.0f - atmB) / t + atmB;
    float G = ((float)inImg[cidx + 1] / 255.0f - atmG) / t + atmG;
    float R = ((float)inImg[cidx + 2] / 255.0f - atmR) / t + atmR;

    outImg[cidx]     = (unsigned char)(fmaxf(fminf(B, 1.0f), 0.0f) * 255.0f);
    outImg[cidx + 1] = (unsigned char)(fmaxf(fminf(G, 1.0f), 0.0f) * 255.0f);
    outImg[cidx + 2] = (unsigned char)(fmaxf(fminf(R, 1.0f), 0.0f) * 255.0f);
}

// ---------------------------------------------------------------------------
// Host operator: orchestrates the three GPU passes and the CPU atm-light step
// ---------------------------------------------------------------------------
__host__ void DehazeFilter::operator()(const unsigned char *input,
                                       unsigned char *output,
                                       unsigned int w, unsigned int h)
{
    unsigned int nPix    = w * h;
    unsigned int bytesIn = nPix * 3;

    prepareBuffers(w, h);

    // upload BGR frame once; dInput is reused by all three GPU passes
    SAFE_CALL(cudaMemcpy(dInput, reinterpret_cast<const void*>(input),
                         bytesIn, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Pass 1: dark channel  (dInput BGR -> dDark 1ch)
    // ------------------------------------------------------------------
    darkChannelKernel<<<this->grid, this->threads>>>(dInput, dDark, w, h, patchHalf);
    SAFE_CALL(cudaDeviceSynchronize());

    // download dark channel to CPU for atmospheric light estimation
    std::vector<unsigned char> darkHost(nPix);
    SAFE_CALL(cudaMemcpy(darkHost.data(), dDark, nPix, cudaMemcpyDeviceToHost));
    computeAtmLight(darkHost.data(), input, w, h);

    // ------------------------------------------------------------------
    // Pass 2: estimate transmission  (dInput BGR -> dTrans 1ch)
    // ------------------------------------------------------------------
    estimateTransmissionKernel<<<this->grid, this->threads>>>(
        dInput, dTrans, w, h,
        atmLight[0], atmLight[1], atmLight[2],
        patchHalf, omega);
    SAFE_CALL(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Pass 3: recover radiance  (dInput + dTrans -> dOutput)
    // ------------------------------------------------------------------
    recoverRadianceKernel<<<this->grid, this->threads>>>(
        dInput, dTrans, dOutput, w, h,
        atmLight[0], atmLight[1], atmLight[2],
        tMin);
    SAFE_CALL(cudaDeviceSynchronize());

    // download dehazed BGR result to host
    SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output), dOutput,
                         bytesIn, cudaMemcpyDeviceToHost));
}
