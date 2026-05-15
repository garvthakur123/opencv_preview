#ifndef SOBELFILTER_HPP_
#define SOBELFILTER_HPP_

#include "ImageFilter.hpp"

/// Implements a sobel operator
class SobelFilter:public ImageFilter {
public:
	SobelFilter(const DeviceInterface&dev, std::string programText,
			std::string kernelName,unsigned int dIn,unsigned int dOut,bool buildKernelFromFile=true) :
			ImageFilter(dev, programText, kernelName, dIn, dOut,buildKernelFromFile) {
	};
	virtual ~SobelFilter() {
	};
	void operator()(unsigned char*input, unsigned char*output, unsigned int currWidth,
			unsigned int currHeight,float c) {
		unsigned int bytesIn = currWidth * currHeight * depthIn;
		unsigned int bytesOut = currWidth * currHeight * depthOut;
		resizeBuffers(currWidth,currHeight);
		// the frame is to be passed as an argument to the kernel function
		// the actual call to the kernel function is represented by an
		// object of type cl::Kernel which stores the kernel arguments
		setArg(0, inputBuffer);
		setArg(1, outputBuffer);

		// image width/height has to be passed anyway for each new input image
		setArg(2, currWidth);
		setArg(3, currHeight);
		setArg(4, c);
		// allocate local memory for the kernel (depends on the work-group
		// size for this kernel; the selected work-group size here is
		// 8+2=10 work-items)
		getKernel().setArg(5,sizeof(unsigned char)*(8+2)*(8+2),NULL);

		// copy frame from CPU to GPU memory
		SAFE_CALL(
				deviceInterface.getQueue().enqueueWriteBuffer(inputBuffer,
						CL_TRUE, 0, bytesIn, input));
		/* The C-way:
		 * cl_int clEnqueueWriteBuffer (cl_command_queue command_queue,
		 *     cl_mem buffer,cl_bool blocking_write,size_t offset,size_t size,
		 *     const void *ptr,cl_uint num_events_in_wait_list,
		 *     const cl_event *event_wait_list,cl_event *event)
		 */

		// compute the number of work-groups, assuming that each work group
		// contains 10*10 workers and produces an 8*8 output pixel patch
		cl::NDRange localRange((8+2),(8+2));
		unsigned int groupsX = (width-1) / 8 + 1;
		unsigned int groupsY = (height-1) / 8 + 1;
		cl::NDRange globalRange(groupsX * (8+2),groupsY * (8+2));
		// call the kernel
		SAFE_CALL(
				deviceInterface.getQueue().enqueueNDRangeKernel(kernel,
						cl::NullRange /* offset */,
						globalRange /* global size (number of work items) */,
						localRange /* work group size (here 10*10) */));
		/*
		 * The C-way:
		 * cl_int clEnqueueNDRangeKernel (cl_command_queue command_queue,cl_kernel kernel,
		 *     cl_uint work_dim,const size_t *global_work_offset,const size_t *global_work_size,
		 *     const size_t *local_work_size,cl_uint num_events_in_wait_list,
		 *     const cl_event *event_wait_list,cl_event *event)
		 *
		 */

		// finish all enqueued commands -- i.e., wait for kernel to finish
		SAFE_CALL(deviceInterface.getQueue().finish());
		/* The C-way:
		 * cl_int clFinish (cl_command_queue command_queue)
		 */

		// copy frame from GPU to CPU memory
		SAFE_CALL(
				deviceInterface.getQueue().enqueueReadBuffer(outputBuffer,
						CL_TRUE, 0, bytesOut, output));
		/*
		 * The C-way:
		 * cl_int clEnqueueReadBuffer (cl_command_queue command_queue,cl_mem buffer,
		 *     cl_bool blocking_read,size_t offset,size_t size,void *ptr,
		 *     cl_uint num_events_in_wait_list,const cl_event *event_wait_list,
		 *     cl_event *event)
		 */
	}

};


#endif /* SOBELFILTER_HPP_ */
