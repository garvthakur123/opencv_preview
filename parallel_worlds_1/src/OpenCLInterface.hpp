#ifndef OPENCL_INTERFACE_HPP_
#define OPENCL_INTERFACE_HPP_

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "cl2.hpp"

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

// ATTENTION: this declaration makes the error checking code unsafe in threaded environments!
static cl_int errorCode;

#include "DeviceInterface.hpp"
#include "ProgramInterface.hpp"

#endif
