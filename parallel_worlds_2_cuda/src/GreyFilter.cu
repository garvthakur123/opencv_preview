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

/// Actual kernel of a grey filter
// @param[in] inImg input image pointer
// @param[out] outImg output image pointer
// @param[in] w width of image
// @param[in] h height of image
template <unsigned int SIZE> __global__ void greyKernel2(const unsigned char*inImg,unsigned char*outImg,
		unsigned int w,unsigned int h){
	unsigned int x = blockIdx.x*SIZE + threadIdx.x;
	unsigned int y = blockIdx.y*SIZE + threadIdx.y;
	__shared__ float3 inCache[SIZE*SIZE];
	constexpr float3 factor={.0722f,0.7152f,.2126f};
	unsigned char result;
    if(y<h) {
        if(x<w) {
        	inCache[SIZE*threadIdx.y+threadIdx.x].x = (float)(((uchar3*)inImg)[x+w*y].x);
        	inCache[SIZE*threadIdx.y+threadIdx.x].y = (float)(((uchar3*)inImg)[x+w*y].y);
        	inCache[SIZE*threadIdx.y+threadIdx.x].z = (float)(((uchar3*)inImg)[x+w*y].z);
            // greyscale conversion (c.f. http://en.wikipedia.org/wiki/Grayscale)
            // Y = 0.2126R + 0.7152G + 0.0722B
        	//inCache.r *= .0722f;
        	//inCache.g *= 0.7152f;
        	//inCache.b *= .2126f;
        	inCache[SIZE*threadIdx.y+threadIdx.x].x = inCache[SIZE*threadIdx.y+threadIdx.x].x * factor.x;
        	inCache[SIZE*threadIdx.y+threadIdx.x].x += inCache[SIZE*threadIdx.y+threadIdx.x].y * factor.y;
        	inCache[SIZE*threadIdx.y+threadIdx.x].x += inCache[SIZE*threadIdx.y+threadIdx.x].z * factor.z;

        	result = (unsigned char)(inCache[SIZE*threadIdx.y+threadIdx.x].x);

        	outImg[x+w*y] = result;
         }
     }
}

__host__ void GreyFilter::operator()(const unsigned char *input,unsigned char *output,
				const unsigned int w, const unsigned int h) {
	this->prepareBuffers(w,h);
	SAFE_CALL(cudaMemcpy(this->dInput,reinterpret_cast<const void*>(input),
				w*h*this->depthIn,cudaMemcpyHostToDevice));
#ifndef SIMPLE_KERNEL
	greyKernel2<BSIZE><<<this->grid,this->threads>>>(dInput,dOutput,w,h);
#else
	greyKernel<<<this->grid,this->threads>>>(dInput,dOutput,w,h);
#endif
	SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output),this->dOutput,
				w*h*this->depthOut,cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaDeviceSynchronize());
}


