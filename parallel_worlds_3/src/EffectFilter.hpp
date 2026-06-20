#ifndef EFFECTFILTER_HPP_
#define EFFECTFILTER_HPP_

#include "SobelFilter.hpp"

/// Implements an effect based on edge detection (SobelColor)
class EffectFilter:public ImageFilter {
protected:
	// image containing edge info
	cl::Buffer edgeBuffer;
	// color depth of edge image in bytes
	unsigned int depthEdge;
public:
	EffectFilter(const DeviceInterface&dev, std::string programText,
			std::string kernelName,unsigned int dIn,unsigned int dEdge,unsigned int dOut,bool buildKernelFromFile=true) :
			ImageFilter(dev,programText,kernelName,dIn,dOut,buildKernelFromFile),depthEdge(dEdge) {};
	virtual ~EffectFilter() {
	};

	void resizeBuffers(unsigned int currWidth,unsigned int currHeight){
		unsigned int bytesEdge = currWidth * currHeight * depthEdge;
		if (currWidth * currHeight > width * height) {
			edgeBuffer = cl::Buffer(deviceInterface.getContext(),
					CL_MEM_READ_ONLY, bytesEdge, NULL,
					&errorCode);
			CHECK_ERROR(errorCode);
		}
		dynamic_cast<ImageFilter*>(this)->resizeBuffers(currWidth,currHeight);
	}
	void operator()(unsigned char*input, unsigned char *edgeInput,
			unsigned char*output, unsigned int currWidth,
			unsigned int currHeight,float threshold) {
		unsigned int bytesIn = currWidth * currHeight * depthIn;
		unsigned int bytesEdge = currWidth * currHeight * depthEdge;
		unsigned int bytesOut = currWidth * currHeight * depthOut;
		resizeBuffers(currWidth,currHeight);
		// the frame, the frame containing edges, and the output frame
		// is to be passed as an argument to the kernel function
		// the actual call to the kernel function is represented by an
		// object of type cl::Kernel which stores the kernel arguments
		setArg(0, outputBuffer);
		setArg(1, edgeBuffer);

		// image width/height/threshold has to be passed anyway for each new input image
		setArg(2, currWidth);
		setArg(3, currHeight);
		setArg(4, threshold);

		// copy frame from CPU to GPU memory
		SAFE_CALL(
				deviceInterface.getQueue().enqueueWriteBuffer(outputBuffer,
						CL_TRUE, 0, (bytesIn<bytesOut)?bytesIn:bytesOut, input));

		SAFE_CALL(
				deviceInterface.getQueue().enqueueWriteBuffer(edgeBuffer,
						CL_TRUE, 0, bytesEdge, edgeInput));

		// call the kernel
		SAFE_CALL(
				deviceInterface.getQueue().enqueueNDRangeKernel(kernel,
						cl::NullRange /* offset */,
						cl::NDRange(width,height) /* global size (number of work items) */,
						cl::NullRange /* work group size (here, again, selected by OpenCL) */));

		// finish all enqueued commands -- i.e., wait for kernel to finish
		SAFE_CALL(deviceInterface.getQueue().finish());

		// copy frame from GPU to CPU memory
		SAFE_CALL(
				deviceInterface.getQueue().enqueueReadBuffer(outputBuffer,
						CL_TRUE, 0, bytesOut, output));
	}
};


#endif /* EFFECTFILTER_HPP_ */
