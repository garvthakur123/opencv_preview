#ifndef _DEVICE_INTERFACE_HPP_
#define _DEVICE_INTERFACE_HPP_

#include <assert.h>

/// representation of platform / device /queue;
/// looks for the first GPU device it can find
// (and settles for a CPU device if none is found) for OpenCL
// computations
class DeviceInterface {
private:

	// vector of all appropriate devices
	VECTOR_CLASS<cl::Device> devices;

	/// a context stores information on device properties that are
	// needed to compile programs for that device (or a group of devices);
	// it is therefore needed to compile an OpenCL program
	cl::Context context;

	/// a queue may be used to submit kernels (functions of
	// compiled OpenCL programs) to a specific device; it is therefore
	// dependent on a certain context and may be used to enqueue a kernel
	// call to one of the (potentially several) devices referenced
	// in the context
	cl::CommandQueue queue;

	/// returns the first devices found of type devType in device;
	// @param devType type of the device to be returned
	// @param deviceVector vector storing the devices
	// @param reference to the device to be returned
	// @returns true, iff a device of the appropriate type
	// could be found
	static bool getFirstDevices(cl_device_type devType,
			VECTOR_CLASS<cl::Device> &deviceVector) {
		VECTOR_CLASS<cl::Platform> platforms;
		// query all platforms
		SAFE_CALL(cl::Platform::get(&platforms));
		/* the C-way:
		 * cl_int clGetPlatformIDs (cl_uint num_entries,
		 * cl_platform_id *platforms,
		 * cl_uint *num_platforms)
		 */
		for (VECTOR_CLASS<cl::Platform>::iterator pit = platforms.begin();
				platforms.end() != pit; ++pit) {
			// try to find a device of type devType
			pit->getDevices(devType, &deviceVector);
			/* the C-way:
			 * cl_int clGetDeviceIDs (cl_platform_id platform,
			 * cl_device_type device_type,
			 * cl_uint num_entries,
			 * cl_device_id *devices,
			 * cl_uint *num_devices)
			 */
			if (deviceVector.size() > 0) {
				return true;
			}
		}
		// no appropriate device found => return false
		return false;
	}

	/// find a GPU device -- if none is found, use a CPU device
	// @returns a vector of all appropriate devices of the same
	// platform (usually, there will be only 1 matching device)
	static VECTOR_CLASS<cl::Device> findDevices
		(cl_device_type requestedDevice=CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR) {
		VECTOR_CLASS<cl::Device> devices;
		// try to get GPU device (otherwise use CPU)
		std::cerr << "Found: ";
		if (getFirstDevices(requestedDevice,
				devices)) {
			std::cerr << "Requested device." << std::endl;
		} else {
			if (getFirstDevices(CL_DEVICE_TYPE_CPU, devices)) {
				std::cerr << "CPU device." << std::endl;
			} else {
				std::cerr << "NO device." << std::endl;
			}
		}
		return devices;
	}

	void init(cl_device_type requestedDevice=CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR) {
		devices = findDevices(requestedDevice);
		// In order to execute a kernel (OpenCL program), it is necessary to first define
		// some hardware presentations:
		// 1. a context that defines the devices (and thus the usable binary code)
		//    that can be used
		// 2. a command queue that manages kernel execution (when to run which kernel)

		// create a context containing only the devices of this platform
		context = cl::Context(devices);
		/* The C-way:
		 * cl_context clCreateContext (const cl_context_properties *properties,
		 * cl_uint num_devices,
		 * const cl_device_id *devices,
		 * void (CL_CALLBACK *pfn_notify)(const char *errinfo,
		 * const void *private_info, size_t cb,
		 * void *user_data),
		 * void *user_data,
		 * cl_int *errcode_ret)
		 */

		// create a command queue for the context -- use the first device found
		queue = cl::CommandQueue(context, this->getDevice(), 0 /* queue properties */,
				&errorCode);
		CHECK_ERROR(errorCode);
		/* The C-way:
		 * cl_command_queue clCreateCommandQueue (cl_context context,
		 * cl_device_id device,
		 * cl_command_queue_properties properties,
		 * cl_int *errcode_ret)
		 */
	}

public:
	DeviceInterface(cl_device_type requestedDevice=CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR) {
		init(requestedDevice);
	}
	DeviceInterface(VECTOR_CLASS<cl::Device> devs,cl::Context ctx,cl::CommandQueue q):devices(devs),
			context(ctx),queue(q){};
	virtual ~DeviceInterface() {
	}

	const cl::Context &getContext() const {
		return context;
	}

	const cl::CommandQueue &getQueue() const {
		return queue;
	}

	/// if a specific device has to be used, always
	// just use the first one here -- of course,
	// one could do better...
	const cl::Device &getDevice(unsigned int i=0) const {
		assert(devices.size()>i);
		return devices[i];
	}
};

#endif
