/*
 * EffectFilter.cu
 *
 * CUDA port of effectFilter.cl from parallel_worlds_3.
 * Darkens color pixels where Sobel edge magnitude exceeds a threshold.
 */

#include "EffectFilter.hpp"

#define ARR(A,x,y,maxX)          (A[(x)+(y)*(maxX)])
#define ARRC(A,x,y,maxX,channel) (A[((x)+(y)*(maxX))*3+(channel)])

/// effect filter kernel
// @param inOutImg  pointer to color image (BGR, 3 bytes/pixel) -- input AND output
// @param edgeImg   pointer to 1-channel Sobel edge map (1 byte/pixel, signed interpretation)
// @param w         image width
// @param h         image height
// @param threshold edge magnitude threshold; pixels above threshold are darkened by 50%
__global__ void effectKernel(unsigned char *inOutImg, const unsigned char *edgeImg,
		unsigned int w, unsigned int h, float threshold) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		float G = (float)((char)ARR(edgeImg, x, y, w));
		float absG = fabsf(G);
		if (absG > threshold) {
			ARRC(inOutImg, x, y, w, 0) = (unsigned char)(ARRC(inOutImg, x, y, w, 0) * 0.5f);
			ARRC(inOutImg, x, y, w, 1) = (unsigned char)(ARRC(inOutImg, x, y, w, 1) * 0.5f);
			ARRC(inOutImg, x, y, w, 2) = (unsigned char)(ARRC(inOutImg, x, y, w, 2) * 0.5f);
		}
	}
}

__host__ void EffectFilter::operator()(const unsigned char *input, const unsigned char *edgeInput,
		unsigned char *output, unsigned int w, unsigned int h, float threshold) {
	this->prepareBuffers(w, h);

	unsigned int bytesIn  = w * h * depthIn;
	unsigned int bytesOut = w * h * depthOut;
	unsigned int bytesEdge = w * h * depthEdge;

	// copy color frame into dOutput (kernel uses it as in-out buffer)
	SAFE_CALL(cudaMemcpy(this->dOutput, reinterpret_cast<const void*>(input),
			(bytesIn < bytesOut) ? bytesIn : bytesOut, cudaMemcpyHostToDevice));

	// copy edge map into dEdge
	SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(this->dEdge),
			reinterpret_cast<const void*>(edgeInput),
			bytesEdge, cudaMemcpyHostToDevice));

	effectKernel<<<this->grid, this->threads>>>(dOutput, dEdge, w, h, threshold);

	SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output), this->dOutput,
			bytesOut, cudaMemcpyDeviceToHost));

	SAFE_CALL(cudaDeviceSynchronize());
}
