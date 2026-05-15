/// Dark Channel Prior dehazing (He et al., CVPR 2009)
// Three kernels: darkChannel -> (CPU: atm light) -> estimateTransmission -> recoverRadiance

// ----- Pass 1 ---------------------------------------------------------------
// Compute the dark channel: for each pixel, find the minimum value across all
// three BGR channels within a (2*patchHalf+1)^2 neighbourhood.
// @param inImg  BGR input, 3 bytes per pixel
// @param darkOut  1-byte-per-pixel output (dark channel map)
// @param w/h  image dimensions
// @param patchHalf  half-size of the patch window (e.g. 7 -> 15x15 window)
__kernel void darkChannel(
    __global unsigned char *inImg,
    __global unsigned char *darkOut,
    unsigned int w, unsigned int h,
    unsigned int patchHalf)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    if (x >= w || y >= h) return;

    unsigned char minVal = 255;
    int ph = (int)patchHalf;
    for (int dy = -ph; dy <= ph; dy++) {
        for (int dx = -ph; dx <= ph; dx++) {
            int nx = clamp((int)x + dx, 0, (int)w - 1);
            int ny = clamp((int)y + dy, 0, (int)h - 1);
            unsigned int idx = ((unsigned int)ny * w + (unsigned int)nx) * 3;
            unsigned char minCh = min(inImg[idx], min(inImg[idx + 1], inImg[idx + 2]));
            minVal = min(minVal, minCh);
        }
    }
    darkOut[y * w + x] = minVal;
}

// ----- Pass 2 ---------------------------------------------------------------
// Estimate transmission map: normalise each pixel by the atmospheric light,
// compute the dark channel of the normalised image, then apply
//   t(x) = 1 - omega * darkChannel(I/A)(x)
// Result is stored as unsigned char (0 = 0.0, 255 = 1.0).
// @param inImg   BGR input, 3 bytes per pixel
// @param transOut  1-byte-per-pixel transmission map
// @param w/h  image dimensions
// @param atmB/atmG/atmR  atmospheric light per channel in [0,1]
// @param patchHalf  same window size as darkChannel pass
// @param omega  haze retention factor (typically 0.95)
__kernel void estimateTransmission(
    __global unsigned char *inImg,
    __global unsigned char *transOut,
    unsigned int w, unsigned int h,
    float atmB, float atmG, float atmR,
    unsigned int patchHalf,
    float omega)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float minVal = 1.0f;
    int ph = (int)patchHalf;
    for (int dy = -ph; dy <= ph; dy++) {
        for (int dx = -ph; dx <= ph; dx++) {
            int nx = clamp((int)x + dx, 0, (int)w - 1);
            int ny = clamp((int)y + dy, 0, (int)h - 1);
            unsigned int idx = ((unsigned int)ny * w + (unsigned int)nx) * 3;
            float normB = (float)inImg[idx]     / (atmB * 255.0f);
            float normG = (float)inImg[idx + 1] / (atmG * 255.0f);
            float normR = (float)inImg[idx + 2] / (atmR * 255.0f);
            minVal = fmin(minVal, fmin(normB, fmin(normG, normR)));
        }
    }
    float t = 1.0f - omega * minVal;
    transOut[y * w + x] = (unsigned char)(clamp(t, 0.0f, 1.0f) * 255.0f);
}

// ----- Pass 3 ---------------------------------------------------------------
// Recover scene radiance:
//   J(x) = (I(x) - A) / max(t(x), tMin) + A
// @param inImg   BGR input, 3 bytes per pixel
// @param transIn  1-byte transmission map from estimateTransmission
// @param outImg  BGR output, 3 bytes per pixel
// @param w/h  image dimensions
// @param atmB/atmG/atmR  atmospheric light per channel in [0,1]
// @param tMin  lower clamp on transmission to avoid division by near-zero
__kernel void recoverRadiance(
    __global unsigned char *inImg,
    __global unsigned char *transIn,
    __global unsigned char *outImg,
    unsigned int w, unsigned int h,
    float atmB, float atmG, float atmR,
    float tMin)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    if (x >= w || y >= h) return;

    unsigned int pidx = y * w + x;
    float t = fmax((float)transIn[pidx] / 255.0f, tMin);

    unsigned int cidx = pidx * 3;
    float B = ((float)inImg[cidx]     / 255.0f - atmB) / t + atmB;
    float G = ((float)inImg[cidx + 1] / 255.0f - atmG) / t + atmG;
    float R = ((float)inImg[cidx + 2] / 255.0f - atmR) / t + atmR;

    outImg[cidx]     = (unsigned char)(clamp(B, 0.0f, 1.0f) * 255.0f);
    outImg[cidx + 1] = (unsigned char)(clamp(G, 0.0f, 1.0f) * 255.0f);
    outImg[cidx + 2] = (unsigned char)(clamp(R, 0.0f, 1.0f) * 255.0f);
}
