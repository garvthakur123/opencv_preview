/*
 * DehazeFilter.cu
 *
 * Dark Channel Prior dehazing — He et al., CVPR 2009.
 *
 * GPU passes:
 *   1. darkChannelKernel          : min BGR over local patch -> dark map
 *   2. histogramKernel            : 256-bin histogram of dark map (shared mem)
 *      atmLightSumKernel          : BGR sums of top-0.1% pixels (atomicAdd)
 *      [CPU: threshold from 1024-byte histogram + atmLight from 32-byte sums]
 *   3. estimateTransmissionKernel : normalize by A, dark channel -> t map
 *   4. recoverRadianceKernel      : J = (I-A)/max(t,tMin) + A -> dehazed BGR
 *
 * PCIe transfers per frame:
 *   Old: ~300 KB (full dark channel downloaded to CPU)
 *   New:   1 KB  (256 histogram bins + 4 sums = 1056 bytes)
 */

#include "DehazeFilter.hpp"

// block size for 1-D kernels (histogram + sum); must be >= 256
static const unsigned int ATM_BLOCK = 256;

// ----- Pass 1 ---------------------------------------------------------------
// Compute the dark channel: for each pixel, find the minimum value across all
// three BGR channels within a (2*patchHalf+1)^2 neighbourhood.
// @param inImg     BGR input, 3 bytes per pixel
// @param darkOut   1-byte-per-pixel dark channel output
// @param w/h       image dimensions
// @param patchHalf half-size of the patch window (e.g. 7 -> 15x15 window)
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

// ----- AtmLight GPU step A --------------------------------------------------
// Build a 256-bin histogram of the dark channel.
// Uses per-block shared memory to minimise global atomicAdd pressure:
// each block accumulates into a private 256-bin histogram in shared memory,
// then merges it into the global histogram with one atomicAdd per bin.
// @param dark   1-byte-per-pixel dark channel (output of pass 1)
// @param hist   global 256-bin unsigned int histogram (pre-cleared to 0)
// @param n      total number of pixels
__global__ void histogramKernel(const unsigned char *dark,
                                unsigned int *hist,
                                unsigned int n)
{
    // private histogram for this block (256 bins, lives in shared memory)
    __shared__ unsigned int localHist[256];
    if (threadIdx.x < 256) localHist[threadIdx.x] = 0;
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&localHist[dark[i]], 1);
    __syncthreads();

    // merge block histogram into global histogram (256 atomicAdds per block)
    if (threadIdx.x < 256)
        atomicAdd(&hist[threadIdx.x], localHist[threadIdx.x]);
}

// ----- AtmLight GPU step B --------------------------------------------------
// Sum the BGR values of every pixel whose dark-channel value >= thresh.
// Results accumulated into sums[0..2] (sumB, sumG, sumR) and sums[3] (count).
// @param dark   1-byte-per-pixel dark channel
// @param color  BGR input, 3 bytes per pixel (same data that dInput holds)
// @param n      total number of pixels
// @param thresh brightness threshold from the histogram step
// @param sums   [sumB, sumG, sumR, count], all unsigned long long (pre-cleared)
__global__ void atmLightSumKernel(const unsigned char *dark,
                                  const unsigned char *color,
                                  unsigned int n,
                                  unsigned char thresh,
                                  unsigned long long *sums)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && dark[i] >= thresh) {
        atomicAdd(&sums[0], (unsigned long long)color[i * 3]);
        atomicAdd(&sums[1], (unsigned long long)color[i * 3 + 1]);
        atomicAdd(&sums[2], (unsigned long long)color[i * 3 + 2]);
        atomicAdd(&sums[3], 1ULL);
    }
}

// ----- Pass 2 ---------------------------------------------------------------
// Estimate transmission map: normalise each neighbour pixel by atmospheric
// light, compute the dark channel of the normalised image, then apply
//   t(x) = 1 - omega * darkChannel(I/A)(x)
// Result stored as unsigned char (0 = 0.0, 255 = 1.0).
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
// Host operator
// ---------------------------------------------------------------------------
__host__ void DehazeFilter::operator()(const unsigned char *input,
                                       unsigned char *output,
                                       unsigned int w, unsigned int h)
{
    unsigned int nPix    = w * h;
    unsigned int bytesIn = nPix * 3;

    prepareBuffers(w, h);

    // upload BGR frame once; dInput is reused by all GPU passes
    SAFE_CALL(cudaMemcpy(dInput, reinterpret_cast<const void*>(input),
                         bytesIn, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Pass 1: dark channel  (dInput BGR -> dDark 1ch)
    // ------------------------------------------------------------------
    darkChannelKernel<<<this->grid, this->threads>>>(dInput, dDark, w, h, patchHalf);
    SAFE_CALL(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // AtmLight — GPU step A: histogram of dark channel
    // Uses 1-D grid (not the 2-D image grid) since dDark is a flat array.
    // ------------------------------------------------------------------
    const unsigned int atmGrid = (nPix + ATM_BLOCK - 1) / ATM_BLOCK;

    SAFE_CALL(cudaMemset(dHist, 0, 256 * sizeof(unsigned int)));
    histogramKernel<<<atmGrid, ATM_BLOCK>>>(dDark, dHist, nPix);
    SAFE_CALL(cudaDeviceSynchronize());

    // Download 256 unsigned ints (1024 bytes) to find threshold on CPU.
    // This is the only large-ish transfer; it replaces the old ~300 KB download.
    unsigned int hostHist[256];
    SAFE_CALL(cudaMemcpy(hostHist, dHist,
                         256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // find brightness threshold = minimum value of the top-0.1% dark pixels
    const unsigned int nTop = std::max(1u, nPix / 1000);
    unsigned int cumCount = 0;
    unsigned char thresh   = 255;
    for (int v = 255; v >= 0; v--) {
        cumCount += hostHist[v];
        if (cumCount >= nTop) { thresh = (unsigned char)v; break; }
    }

    // ------------------------------------------------------------------
    // AtmLight — GPU step B: sum BGR of qualifying pixels
    // ------------------------------------------------------------------
    SAFE_CALL(cudaMemset(dSums, 0, 4 * sizeof(unsigned long long)));
    atmLightSumKernel<<<atmGrid, ATM_BLOCK>>>(dDark, dInput, nPix, thresh, dSums);
    SAFE_CALL(cudaDeviceSynchronize());

    // Download 4 unsigned long longs (32 bytes) and compute atmLight.
    // All arithmetic here is 3 divisions + 6 clamps — negligible CPU work.
    unsigned long long hostSums[4];
    SAFE_CALL(cudaMemcpy(hostSums, dSums,
                         4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    unsigned long long cnt = std::max(1ULL, hostSums[3]);
    const float capVal = 0.85f;
    atmLight[0] = std::min(std::max((float)hostSums[0] / (cnt * 255.0f), 1.0f / 255.0f), capVal);
    atmLight[1] = std::min(std::max((float)hostSums[1] / (cnt * 255.0f), 1.0f / 255.0f), capVal);
    atmLight[2] = std::min(std::max((float)hostSums[2] / (cnt * 255.0f), 1.0f / 255.0f), capVal);

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
