/*
 * GreyFilter.hpp
 *
 *  Created on: 06.12.2021
 *      Author: faber
 */

#ifndef SRC_GREYFILTER_HPP_
#define SRC_GREYFILTER_HPP_

#include "ImageFilter.hpp"



/// This class represents a kernel used as a grey filter as a functor
// it takes pointers to input and output images, width and height as
// parameters in its operator and wraps the kernel launch
//
class GreyFilter: public ImageFilter {
public:
	GreyFilter(unsigned int dIn,unsigned int dOut):
		ImageFilter(dIn,dOut){};
	void operator()(const unsigned char *input,unsigned char *output,
					const unsigned int w, const unsigned int h);
};



#endif /* SRC_GREYFILTER_HPP_ */
