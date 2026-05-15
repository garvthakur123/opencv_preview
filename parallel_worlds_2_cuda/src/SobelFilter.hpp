#ifndef SOBELFILTER_HPP_
#define SOBELFILTER_HPP_

#include "ImageFilter.hpp"

/// Implements a sobel operator
class SobelFilter:public ImageFilter {
public:
	SobelFilter(unsigned int dIn,unsigned int dOut) :
		ImageFilter(dIn, dOut) {
	};
	virtual void resizeGrid(unsigned int w, unsigned int h){
		this->threads = dim3(BSIZE,BSIZE,1);
		this->grid =
				dim3((w+(this->threads.x-2)-1)/(this->threads.x-2),
					(h+(this->threads.y-2)-1)/(this->threads.y-2));
	}
	virtual ~SobelFilter() {
	};
	void operator()(const unsigned char*input, unsigned char*output, const unsigned int currWidth,
			const unsigned int currHeight,const float c);
};


#endif /* SOBELFILTER_HPP_ */
