/*
 * GreyFilter.cu
 *
 *  Created on: 06.12.2021
 *      Author: faber
 */

#include "GreyFilter.hpp"


/// Actual kernel of a grey filter
// @param[in] inImg input image pointer
// @param[out] outImg output image pointer
// @param[in] w width of image
// @param[in] h height of image
__global__ void greyKernel(unsigned const char*inImg,unsigned char*outImg,
		unsigned int w,unsigned int h){
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(y<h) {
        if(x<w) {
            // greyscale conversion (c.f. http://en.wikipedia.org/wiki/Grayscale)
            // Y = 0.2126R + 0.7152G + 0.0722B
            outImg[x+w*y] = 0.0722 * inImg[3*(x+w*y)] /* blue */
                + 0.7152 * inImg[3*(x+w*y)+1]             /* green */
                + 0.2126 * inImg[3*(x+w*y)+2];            /* red */
         }
     }
}


__host__ void GreyFilter::operator()(const unsigned char *input,unsigned char *output,
				const unsigned int w, const unsigned int h) {
	this->prepareBuffers(w,h);
	SAFE_CALL(cudaMemcpy(this->dInput,reinterpret_cast<const void*>(input),
				w*h*this->depthIn,cudaMemcpyHostToDevice));
	greyKernel<<<this->grid,this->threads>>>(dInput,dOutput,w,h);
	SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output),this->dOutput,
				w*h*this->depthOut,cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaDeviceSynchronize());
}


