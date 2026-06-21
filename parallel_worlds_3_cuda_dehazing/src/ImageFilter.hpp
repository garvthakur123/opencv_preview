#ifndef _IMAGE_FILTER_HPP_
#define _IMAGE_FILTER_HPP_

#include "CUDAInterface.hpp"

// block size
constexpr unsigned int BSIZE = 8;

/// A kernel that has one image as
// input and one as output; the kernel for this image filter
// has to take exactly 4 arguments:
// 1. pointer to the input image data
// 2. pointer to the output image data
// 3. image width
// 4. image height
// the kernel is then called once for each pixel of the image
// (and needs to know the data format by itself)
class ImageFilter {
protected:
	/// pointer to the input image data on the GPU
	unsigned char *dInput;
	/// pointer to the output image data
	unsigned char *dOutput;
	/// image width
	unsigned int width;
	/// image height
	unsigned int height;
	/// color depth in bytes (input)
	unsigned int depthIn;
	/// color depth in bytes (output)
	unsigned int depthOut;
	/// grid extent
	dim3 grid;
	/// block extent
	dim3 threads;
public:
	ImageFilter(unsigned int dIn,unsigned int dOut) :
		dInput(nullptr),dOutput(nullptr),
			width(0), height(0),depthIn(dIn),depthOut(dOut){
		this->resizeGrid(width,height);
	}
	virtual ~ImageFilter() {
		SAFE_CALL(cudaFree(dInput));
		SAFE_CALL(cudaFree(dOutput));
	}

	/// resize CUDA grid structure according to image width/height
	virtual void resizeGrid(unsigned int w, unsigned int h){
		// greyFilter kernel 2 needs size.x == size.y !!
		this->threads = dim3(BSIZE,BSIZE);
		this->grid =
				dim3((w+this->threads.x-1)/this->threads.x,
					(h+this->threads.y-1)/this->threads.y);
	}


	/// resizes dInput/dOutput so that each can take a color image
	// of the specified width & height
	virtual void resizeBuffers(unsigned int currWidth,unsigned int currHeight) {
		unsigned int bytesIn = currWidth * currHeight * depthIn;
		unsigned int bytesOut = currWidth * currHeight * depthOut;
		this->resizeGrid(currWidth, currHeight);
		// if the image is larger than input/output buffers,
		// create new buffers
		if (currWidth * currHeight > width * height) {
			SAFE_CALL(cudaFree(dInput));
			SAFE_CALL(cudaFree(dOutput));
			SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dInput), bytesIn));
			SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dOutput), bytesOut));
			width = currWidth;
			height = currHeight;
		}

	}

	virtual void prepareBuffers(unsigned int currWidth,unsigned int currHeight) {
		resizeBuffers(currWidth,currHeight);
	}


};

#endif

