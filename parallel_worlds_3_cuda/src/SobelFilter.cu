/*
 * GreyFilter.cu
 *
 *  Created on: 06.12.2021
 *      Author: faber
 */

#include "SobelFilter.hpp"


#define ARR(A,x,y,maxX) (A[(x)+(y)*(maxX)])

constexpr unsigned int SIZE = BSIZE + 2;

        /// sobel filter (cf. http://en.wikipedia.org/wiki/Sobel_operator):
        // detect edges by computing the convolution
        // with matrix {{-1,0,1},{-2,0,2},{-1,0,1}} in x- and y- direction;
        // the result is computed as c*sqrt(G_x^2 + G_y^2) (where G_x/G_y
        // is the convolution with the above matrix);
        // this computation is only done for interior pixels -- the edges
        // of the image are blacked out;
        // @param inImg pointer to the input grey image in device memory
        // @param outImg pointer to the output grey image in device memory
        // @param w width of image
        // @param h height of image
        // @param c coefficient by which to multiply the actual convolution
        // @param img image portion for computation -- to be shared between
        //          work-items of a work-group (each work-item writes exactly
        //          1 pixel of img)
        // Note: img has to be passed via Kernel::setArg(), because its size
        // depends on the size of the work-group (otherwise it could have been
        // defined inside the kernel)
__global__ void sobelKernel(const unsigned char*inImg,unsigned char*outImg,
		const unsigned int w,const unsigned int h, const float c){
	__shared__ float cache[SIZE*SIZE];
    // coordinates of input pixel in cache array img
    unsigned int xCache;unsigned  int yCache;
    // coordinates of pixel in input/output image
    unsigned int x;unsigned  int y;
    // number of output pixels per work-group in x/y direction
    // will evaluate to 8, since the kernel will be started on a
    // 10 * 10 work-group
    unsigned int numOutX; unsigned int numOutY;
    numOutX = blockDim.x - 2; numOutY = blockDim.y - 2;
    x = blockIdx.x * numOutX + threadIdx.x;
    y = blockIdx.y * numOutY + threadIdx.y;
    xCache = threadIdx.x; yCache = threadIdx.y;
    if(x<w && y<h){
        // read pixels from original image into cache
         ARR(cache,xCache,yCache,blockDim.x) = ARR(inImg,x,y,w);
         // border pixels are all black
         if(0==x||0==y||w-1==x||h-1==y){
             ARR(outImg,x,y,w) = 0;
         }
         // wait for all work-items to finish copying
          __syncthreads();
          if(x<w-1 && y<h-1){
             // compute result value and write it back to device memory
             // (but only for interior pixels, i.e. 1<=id<=max-1)
             if(xCache > 0 && xCache < blockDim.x - 1){
                 if(yCache > 0 && yCache < blockDim.y - 1){
                     float G_x =
                                 -ARR(cache,xCache-1,yCache-1,blockDim.x)
                                 -2*ARR(cache,xCache-1,yCache,blockDim.x)
                                 -ARR(cache,xCache-1,yCache+1,blockDim.x)
                                 +ARR(cache,xCache+1,yCache-1,blockDim.x)
                                 +2*ARR(cache,xCache+1,yCache,blockDim.x)
                                 +ARR(cache,xCache+1,yCache+1,blockDim.x);
                     float G_y =
                                 -ARR(cache,xCache-1,yCache-1,blockDim.x)
                                 -2*ARR(cache,xCache,yCache-1,blockDim.x)
                                 -ARR(cache,xCache+1,yCache-1,blockDim.x)
                                 +ARR(cache,xCache-1,yCache+1,blockDim.x)
                                 +2*ARR(cache,xCache,yCache+1,blockDim.x)
                                 +ARR(cache,xCache+1,yCache+1,blockDim.x);
                     // sqrt is a predefined OpenCL function!
                     ARR(outImg,x,y,w) = (unsigned char) (c * sqrt(G_x*G_x + G_y*G_y));
                 }
           }
        }

    }
}

__host__ void SobelFilter::operator()(const unsigned char *input,unsigned char *output,
				const unsigned int w, const unsigned int h, const float c) {
	this->prepareBuffers(w,h);
	SAFE_CALL(cudaMemcpy(this->dInput,reinterpret_cast<const void*>(input),
				w*h*this->depthIn,cudaMemcpyHostToDevice));
	sobelKernel<<<this->grid,this->threads>>>(dInput,dOutput,w,h,c);
	SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output),this->dOutput,
				w*h*this->depthOut,cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaDeviceSynchronize());
}


