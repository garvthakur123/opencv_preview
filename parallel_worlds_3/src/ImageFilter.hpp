#ifndef _IMAGE_FILTER_HPP_
#define _IMAGE_FILTER_HPP_

#include "ProgramInterface.hpp"

/// A kernel that has one image as
// input and one as output; the kernel for this image filter
// has to take exactly 4 arguments:
// 1. pointer to the input image data
// 2. pointer to the output image data
// 3. image width
// 4. image height
// the kernel is then called once for each pixel of the image
// (and needs to know the data format by itself)
class ImageFilter: public ProgramInterface {
protected:
	/// pointer to the input image data
	cl::Buffer inputBuffer;
	/// pointer to the output image data
	cl::Buffer outputBuffer;
	/// image width
	unsigned int width;
	/// image height
	unsigned int height;
	/// color depth in bytes (input)
	unsigned int depthIn;
	/// color depth in bytes (output)
	unsigned int depthOut;
public:
	ImageFilter(const DeviceInterface&dev, std::string programText,
			std::string kernelName,unsigned int dIn,unsigned int dOut, bool buildKernelFromFile=true) :
			ProgramInterface(dev, programText, kernelName,buildKernelFromFile), width(0), height(0),
			depthIn(dIn),depthOut(dOut){
	}
	;
	virtual ~ImageFilter() {
	}
	;
	/// resizes inputBuffer/outputBuffer so that each can take a color image
	// of the specified width & height
	void resizeBuffers(unsigned int currWidth,unsigned int currHeight) {
		unsigned int bytesIn = currWidth * currHeight * depthIn;
		unsigned int bytesOut = currWidth * currHeight * depthOut;
		// if the image is larger than input/output buffers,
		// create new buffers
		if (currWidth * currHeight > width * height) {
			inputBuffer = cl::Buffer(deviceInterface.getContext(),
					CL_MEM_READ_ONLY, bytesIn, NULL,
					&errorCode);
			CHECK_ERROR(errorCode);
			/* The C-way:
			 * cl_mem clCreateBuffer (cl_context context,
			 * cl_mem_flags flags,
			 * size_t size,
			 * void *host_ptr,
			 * cl_int *errcode_ret)
			 */
			outputBuffer = cl::Buffer(deviceInterface.getContext(),
					CL_MEM_WRITE_ONLY, bytesOut, NULL,
					&errorCode);
			CHECK_ERROR(errorCode);

			width = currWidth;
			height = currHeight;
		}

	}
	void operator()(unsigned char*input, unsigned char*output, unsigned int currWidth,
			unsigned int currHeight) {
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

		// call the kernel
		SAFE_CALL(
				deviceInterface.getQueue().enqueueNDRangeKernel(kernel,
						cl::NullRange /* offset */,
						cl::NDRange(width,
								height) /* global size (number of work items) */,
						cl::NullRange /* work group size (here selected by OpenCL) */));
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


#endif
