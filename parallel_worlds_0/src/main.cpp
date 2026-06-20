//============================================================================
// Name        : DeviceQuery.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
//============================================================================


#include <iostream>
#include <sstream>
#include <stdlib.h>

#include "cl2.hpp"

/*
 * VECTOR_CLASS and STRING_CLASS are defined by OpenCL so that
 * the user may use her own implementation of these classes, if she
 * wants to avoid the STL; the defines below are only safeguards to have
 * all these definitions available
 */
#ifndef STRING_CLASS
#define STRING_CLASS ::std::string
//cl::STRING_CLASS
#endif
#ifndef VECTOR_CLASS
#define VECTOR_CLASS ::std::vector
//cl::VECTOR_CLASS
#endif

/*
 * Defines for use in OpenCL functions that return an error code;
 * these defines query the return code and produce an exception, if
 * an error occurs
 */
#define STRG(y) (#y)
// STRING can be used to create an error message and to produce
// a program text for OpenCL kernels
#define STRING(x) (STRING_CLASS(STRG(x)))

#define CL_ERROR(fct,line,errorCode) do{std::stringstream ss;ss << STRING_CLASS("FAILED CALL (at '")+ __FILE__ +"', line "+STRING(__LINE__)+"): "+(#fct)+" failed with "<<errorCode<<"."; std::cerr<<ss.str()<<std::endl;exit(1);}while(0)
#define SAFE_CALL(fct_call) do{cl_int errC = fct_call; if(CL_SUCCESS!=(errC)){CL_ERROR(fct_call,__LINE__,errC);}}while(0)
#define CHECK_ERROR(errCode) do{SAFE_CALL(errCode);}while(0)

cl_int errorCode;

int main() {
	VECTOR_CLASS<cl::Platform> platforms;

	// query all platforms
	SAFE_CALL(cl::Platform::get(&platforms));
	/* the C-way:
	 * cl_int clGetPlatformIDs (cl_uint num_entries,
	 * cl_platform_id *platforms,
	 * cl_uint *num_platforms)
	 */

	int i;
	for(VECTOR_CLASS<cl::Platform>::iterator pit=platforms.begin();
			platforms.end()!=pit;++pit) {
		std::cout<<"===============\n";

		std::pair<cl_platform_info,std::string> oclPlatformInfos[]=
		{	std::make_pair(CL_PLATFORM_VENDOR,"CL_PLATFORM_VENDOR"),
			std::make_pair(CL_PLATFORM_NAME,"CL_PLATFORM_NAME"),
			std::make_pair(CL_PLATFORM_VERSION,"CL_PLATFORM_VERSION"),
			std::make_pair(CL_PLATFORM_PROFILE,"CL_PLATFORM_PROFILE"),
			std::make_pair(CL_PLATFORM_EXTENSIONS,"CL_PLATFORM_EXTENSIONS"),
			std::make_pair(0,"")};

		// print info for current platform
		for(i=0;0!=oclPlatformInfos[i].first;++i) {
			std::string value;
			CHECK_ERROR(pit->getInfo(oclPlatformInfos[i].first,&value));
			/* the C-way:
			 * cl_int clGetPlatformInfo (cl_platform_id platform,
			 * cl_platform_info param_name,
			 * size_t param_value_size,
			 * void *param_value,
			 * size_t *param_value_size_ret)
			 */
			std::cout<<oclPlatformInfos[i].second<<": "<<value<<"\n";
		}

		// query GPU/accelerator devices
		VECTOR_CLASS<cl::Device> devices;
		errorCode = pit->getDevices
		(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, &devices);
		/* the C-way:
		 * cl_int clGetDeviceIDs (cl_platform_id platform,
		 * cl_device_type device_type,
		 * cl_uint num_entries,
		 * cl_device_id *devices,
		 * cl_uint *num_devices)
		 */

		// print info for each device of the current platform
		for(VECTOR_CLASS<cl::Device>::iterator dev=devices.begin();
				devices.end()!=dev;++dev) {
			std::pair<cl_device_info,std::string> oclDeviceInfos[]=
			{	std::make_pair(CL_DEVICE_MAX_COMPUTE_UNITS,"CL_DEVICE_MAX_COMPUTE_UNITS"),
				std::make_pair(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,"CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS"),
				std::make_pair(CL_DEVICE_MAX_WORK_GROUP_SIZE,"CL_DEVICE_MAX_WORK_GROUP_SIZE"),
				std::make_pair(CL_DEVICE_MAX_CLOCK_FREQUENCY,"CL_DEVICE_MAX_CLOCK_FREQUENCY"),
				std::make_pair(0,"")};

			std::cout<<"---------------\n";
			for(i=0;0!=oclDeviceInfos[i].first;++i) {
				cl_uint value;
				cl_int errC = dev->getInfo(oclDeviceInfos[i].first,&value);
				/* the C-way:
				 * cl_int clGetDeviceInfo (cl_device_id device,
				 * cl_device_info param_name,
				 * size_t param_value_size,
				 * void *param_value,
				 * size_t *param_value_size_ret)
				 */
				if(CL_SUCCESS != errC){
					std::cout<<oclDeviceInfos[i].second<<": N/A\n";
					continue;
				}
				std::cout<<oclDeviceInfos[i].second<<": "<<value<<"\n";
			}
			std::cout<<"---------------\n";

		}
		std::cout<<"===============\n";
	}
}
