#ifndef _PROGRAM_INTERFACE_HPP_
#define _PROGRAM_INTERFACE_HPP_

#include <string>
#include <fstream>
#include <algorithm>

#include "DeviceInterface.hpp"

class ProgramInterface {
	std::string rot13(std::string &s){
		std::for_each(s.begin(),s.end(),[](char &c){c=(!isalpha(c))?c:((c>'Z')?('a'+(c-'a'+13)%26):('A'+(c-'A'+13)%26));});return s;
	}
protected:
	/// description of the platform/device/queue to be used
	DeviceInterface deviceInterface;

	/// each kernel invocation is represented by an object of type kernel
	cl::Kernel kernel;

public:
	/// contructor needs description of the hardware
	// in order to be able to compile programs for it
	ProgramInterface(const DeviceInterface&dev, std::string programText,
			std::string kernelName, bool buildKernelFromFile=true) :
			deviceInterface(dev) {
		if(buildKernelFromFile){
			kernelFromFile(programText, kernelName);
		} else {
			kernelFromString(programText, kernelName);
		}
	}
	virtual ~ProgramInterface() {
	}

	cl::Program buildProgram(std::string programText) {
		if(programText.find("// rot13") != std::string::npos){
			rot13(programText);
		}
		// create a program object and immediately compile the program text for it (build=true)
		// into a binary form that can be executed on the devices of the given context;
		cl::Program program(deviceInterface.getContext(), programText, true,
				&errorCode);
		/* The C-way:
		 * creating a program object:
		 * cl_program clCreateProgramWithSource (cl_context context,
		 * cl_uint count,
		 * const char **strings,
		 * const size_t *lengths,
		 * cl_int *errcode_ret)
		 * building a program (compile & link -- can also be done separately):
		 * cl_int clBuildProgram (cl_program program,
		 * cl_uint num_devices,
		 * const cl_device_id *device_list,
		 * const char *options,
		 * void (CL_CALLBACK *pfn_notify)(cl_program program,
		 * void *user_data),
		 * void *user_data)
		 */

		// if the compilation failed, get a more precise error message
		if (CL_BUILD_PROGRAM_FAILURE == errorCode) {
			std::string buildInfo = "";
			buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
					deviceInterface.getDevice());
			/*
			 * The C-way:
			 * cl_int clGetProgramBuildInfo (cl_program program,cl_device_id device,
			 *     cl_program_build_info param_name,size_t param_value_size,
			 *     void *param_value,size_t *param_value_size_ret)
			 */
			std::cerr<<"OpenCL build error:\n"<<buildInfo<<std::endl;
		}
		CHECK_ERROR(errorCode);
		return program;
	}

	/// creating an actual kernel program consists of
	// 1. creating a program object
	// 2. building a program by
	//    a. compiling some program text for use on the device(s)
	//    b. linking the program (e.g. with library functions)
	// in C++, these steps can all be performed simply by calling the
	// Program constructor
	void kernelFromString(std::string programText, std::string kernelName) {
		cl::Program program = buildProgram(programText);
		// the frame is to be passed as an argument to the kernel function
		// the actual call to the kernel function is represented by an
		// object of type cl::Kernel which stores the kernel arguments
		kernel = cl::Kernel(program, kernelName.c_str(), &errorCode);
		CHECK_ERROR(errorCode);
		/* The C-way:
		 * cl_kernel clCreateKernel (cl_program program,
		 * const char *kernel_name,
		 * cl_int *errcode_ret)
		 */
	}

	/// construct a kernel reading the program text from a file
	// (has to be placed in the cwd -- for eclipse usually the project directory)
	void kernelFromFile(std::string programTextFileName,std::string kernelName){
		std::ifstream inputStream(programTextFileName.c_str());
		if(!inputStream){
			std::cerr<<"could not open file "<<programTextFileName<<" -- wrong CWD?"<<std::endl;
			exit(1);
		}
		std::string programText = "";
		std::string programLine;
		while(std::getline(inputStream,programLine)){
			programText += programLine + "\n";
		}
		kernelFromString(programText,kernelName);
	}


	cl::Kernel &getKernel() {
		return kernel;
	}

	template<typename T> void setArg(unsigned int num, T&argument) {
		SAFE_CALL(kernel.setArg(num, argument));
		/* The C-way:
		 * cl_int clSetKernelArg (cl_kernel kernel,
		 * cl_uint arg_index,
		 * size_t arg_size,
		 * const void *arg_value)
		 */
	}

	void setArg(unsigned int num, size_t arraySize, void *arrayPtr) {
		SAFE_CALL(kernel.setArg(num, arraySize, arrayPtr));
		/* The C-way:
		 * cl_int clSetKernelArg (cl_kernel kernel,
		 * cl_uint arg_index,
		 * size_t arg_size,
		 * const void *arg_value)
		 */
	}

};

#endif /* PROGRAMINTERFACE_HPP_ */
