#ifndef CUDA_INTERFACE_HPP_
#define CUDA_INTERFACE_HPP_

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "helper_cuda.h"

/*
 * VECTOR_CLASS and STRING_CLASS are defined by OpenCL 1.x BUT NOT 2.x(!)
 * so that the user may use her own implementation of these classes, if she
 * wants to avoid the STL; the defines below are only safeguards to have
 * all these definitions available
 */
#ifndef STRING_CLASS
#define STRING_CLASS ::std::string
#endif
#ifndef VECTOR_CLASS
#define VECTOR_CLASS ::std::vector
#endif

/*
 * Defines for use in CUDA functions that return an error code;
 * these actually already exists in CUDA helpers :-)
 * these defines query the return code and produce an exception, if
 * an error occurs
 */
#define SAFE_CALL(fct_call) checkCudaErrors(fct_call)


#endif
