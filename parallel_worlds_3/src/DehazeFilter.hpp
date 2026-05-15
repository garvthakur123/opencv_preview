#ifndef DEHAZEFILTER_HPP_
#define DEHAZEFILTER_HPP_

#include <vector>
#include <algorithm>
#include <fstream>

#include "ImageFilter.hpp"

/// Implements Dark Channel Prior dehazing (He et al., CVPR 2009) as a
/// three-pass OpenCL pipeline:
///   1. darkChannel kernel  (GPU)  -> 1-ch dark map
///   2. computeAtmLight            (CPU) -> atmospheric light A[3]
///   3. estimateTransmission kernel (GPU) -> 1-ch transmission map
///   4. recoverRadiance kernel      (GPU) -> BGR dehazed output
///
/// Follows the same class pattern as EffectFilter: subclass of ImageFilter
/// with additional internal cl::Buffers and extra cl::Kernel objects built
/// from the same .cl file.
class DehazeFilter : public ImageFilter {
    // extra kernels for passes 2 & 3
    cl::Kernel transmissionKernel;
    cl::Kernel radianceKernel;

    // intermediate device buffers
    cl::Buffer darkBuffer;   // 1 byte per pixel  (dark channel map)
    cl::Buffer transBuffer;  // 1 byte per pixel  (transmission map)

    // atmospheric light per BGR channel in [0,1], updated each frame
    float atmLight[3];

    // tuning parameters
    unsigned int patchHalf; // half-side of the min-filter patch (default 7 -> 15x15)
    float omega;            // haze-retention factor (default 0.95)
    float tMin;             // minimum transmission clamp (default 0.1)

    /// Estimate atmospheric light on the CPU from the dark channel and the
    /// original colour frame.  Top 0.1 % of dark-channel pixels are
    /// candidates; A is the per-channel mean of all candidates, capped to
    /// prevent a single bright highlight (e.g. a window) from dominating
    /// and causing extreme colour shifts on non-hazy scenes.
    void computeAtmLight(const unsigned char* dark,
                         const unsigned char* color,
                         unsigned int w, unsigned int h)
    {
        unsigned int n = w * h;
        unsigned int nTop = std::max(1u, n / 1000);

        // find the threshold value using partial sort
        std::vector<unsigned char> vals(dark, dark + n);
        std::nth_element(vals.begin(), vals.begin() + (n - nTop), vals.end());
        unsigned char thresh = vals[n - nTop];

        // accumulate per-channel mean across all qualifying candidates;
        // using mean instead of the single brightest pixel makes the estimate
        // far more robust when a coloured highlight (window, lamp) is present
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

        // clamp each channel: floor at 1/255 (avoid divide-by-zero in kernel),
        // ceiling at 0.85 (avoid extreme subtraction on non-hazy scenes)
        const float capVal = 0.85f;
        atmLight[0] = std::min(std::max(sumB / (count * 255.0f), 1.0f / 255.0f), capVal);
        atmLight[1] = std::min(std::max(sumG / (count * 255.0f), 1.0f / 255.0f), capVal);
        atmLight[2] = std::min(std::max(sumR / (count * 255.0f), 1.0f / 255.0f), capVal);
    }

public:
    /// @param dev          shared OpenCL device (same as all other filters)
    /// @param programFile  path to dehazeFilters.cl (runtime-relative, e.g. "src/dehazeFilters.cl")
    /// @param patchHalf_   half-side of the DCP patch window (default 7)
    /// @param omega_       haze retention factor; lower = less aggressive (default 0.75)
    /// @param tMin_        minimum transmission clamp; higher = less amplification (default 0.2)
    DehazeFilter(const DeviceInterface& dev,
                 std::string programFile,
                 unsigned int patchHalf_ = 7,
                 float omega_ = 0.75f,
                 float tMin_  = 0.2f)
        : ImageFilter(dev, programFile, "darkChannel", 3, 3, true),
          patchHalf(patchHalf_), omega(omega_), tMin(tMin_)
    {
        // The base class already built the program for "darkChannel".
        // Re-read the same file to build the other two kernels from it.
        std::ifstream f(programFile.c_str());
        if (!f) {
            std::cerr << "DehazeFilter: could not open " << programFile << std::endl;
            exit(1);
        }
        std::string src((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        cl::Program program = buildProgram(src);

        transmissionKernel = cl::Kernel(program, "estimateTransmission", &errorCode);
        CHECK_ERROR(errorCode);
        radianceKernel = cl::Kernel(program, "recoverRadiance", &errorCode);
        CHECK_ERROR(errorCode);

        atmLight[0] = atmLight[1] = atmLight[2] = 1.0f;
    }

    virtual ~DehazeFilter() {}

    /// Allocate (or grow) all device buffers for the given image size.
    void resizeBuffers(unsigned int currWidth, unsigned int currHeight) {
        if (currWidth * currHeight > width * height) {
            unsigned int nPix = currWidth * currHeight;
            darkBuffer = cl::Buffer(deviceInterface.getContext(),
                                    CL_MEM_READ_WRITE, nPix, NULL, &errorCode);
            CHECK_ERROR(errorCode);
            transBuffer = cl::Buffer(deviceInterface.getContext(),
                                     CL_MEM_READ_WRITE, nPix, NULL, &errorCode);
            CHECK_ERROR(errorCode);
        }
        // allocates inputBuffer (3ch) and outputBuffer (3ch), updates width/height
        ImageFilter::resizeBuffers(currWidth, currHeight);
    }

    /// Run the full dehazing pipeline.
    /// Signature matches ImageFilter::operator() so usage in main.cpp is identical.
    void operator()(unsigned char* input, unsigned char* output,
                    unsigned int currWidth, unsigned int currHeight)
    {
        unsigned int nPix    = currWidth * currHeight;
        unsigned int bytesIn = nPix * 3;

        resizeBuffers(currWidth, currHeight);

        // upload original BGR once; reused by all three GPU passes
        SAFE_CALL(deviceInterface.getQueue().enqueueWriteBuffer(
            inputBuffer, CL_TRUE, 0, bytesIn, input));

        // ------------------------------------------------------------------
        // Pass 1: dark channel  (inputBuffer BGR -> darkBuffer 1ch)
        // ------------------------------------------------------------------
        SAFE_CALL(kernel.setArg(0, inputBuffer));
        SAFE_CALL(kernel.setArg(1, darkBuffer));
        SAFE_CALL(kernel.setArg(2, currWidth));
        SAFE_CALL(kernel.setArg(3, currHeight));
        SAFE_CALL(kernel.setArg(4, patchHalf));
        SAFE_CALL(deviceInterface.getQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange,
            cl::NDRange(currWidth, currHeight), cl::NullRange));
        SAFE_CALL(deviceInterface.getQueue().finish());

        // read dark channel back to CPU for atmospheric light estimation
        std::vector<unsigned char> darkHost(nPix);
        SAFE_CALL(deviceInterface.getQueue().enqueueReadBuffer(
            darkBuffer, CL_TRUE, 0, nPix, darkHost.data()));
        computeAtmLight(darkHost.data(), input, currWidth, currHeight);

        // ------------------------------------------------------------------
        // Pass 2: estimate transmission  (inputBuffer BGR -> transBuffer 1ch)
        // ------------------------------------------------------------------
        SAFE_CALL(transmissionKernel.setArg(0, inputBuffer));
        SAFE_CALL(transmissionKernel.setArg(1, transBuffer));
        SAFE_CALL(transmissionKernel.setArg(2, currWidth));
        SAFE_CALL(transmissionKernel.setArg(3, currHeight));
        SAFE_CALL(transmissionKernel.setArg(4, atmLight[0]));
        SAFE_CALL(transmissionKernel.setArg(5, atmLight[1]));
        SAFE_CALL(transmissionKernel.setArg(6, atmLight[2]));
        SAFE_CALL(transmissionKernel.setArg(7, patchHalf));
        SAFE_CALL(transmissionKernel.setArg(8, omega));
        SAFE_CALL(deviceInterface.getQueue().enqueueNDRangeKernel(
            transmissionKernel, cl::NullRange,
            cl::NDRange(currWidth, currHeight), cl::NullRange));
        SAFE_CALL(deviceInterface.getQueue().finish());

        // ------------------------------------------------------------------
        // Pass 3: recover radiance  (inputBuffer + transBuffer -> outputBuffer)
        // ------------------------------------------------------------------
        SAFE_CALL(radianceKernel.setArg(0, inputBuffer));
        SAFE_CALL(radianceKernel.setArg(1, transBuffer));
        SAFE_CALL(radianceKernel.setArg(2, outputBuffer));
        SAFE_CALL(radianceKernel.setArg(3, currWidth));
        SAFE_CALL(radianceKernel.setArg(4, currHeight));
        SAFE_CALL(radianceKernel.setArg(5, atmLight[0]));
        SAFE_CALL(radianceKernel.setArg(6, atmLight[1]));
        SAFE_CALL(radianceKernel.setArg(7, atmLight[2]));
        SAFE_CALL(radianceKernel.setArg(8, tMin));
        SAFE_CALL(deviceInterface.getQueue().enqueueNDRangeKernel(
            radianceKernel, cl::NullRange,
            cl::NDRange(currWidth, currHeight), cl::NullRange));
        SAFE_CALL(deviceInterface.getQueue().finish());

        // read dehazed BGR result back to host
        SAFE_CALL(deviceInterface.getQueue().enqueueReadBuffer(
            outputBuffer, CL_TRUE, 0, bytesIn, output));
    }
};

#endif /* DEHAZEFILTER_HPP_ */
