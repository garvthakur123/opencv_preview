#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CUDAInterface.hpp"
#include "GreyFilter.hpp"
#include "SobelFilter.hpp"

GreyFilter greyFilter(3,1);
SobelFilter sobelFilter(1, 1);


__host__ int main(int argc, const char** argv) {
	cv::VideoCapture capture(0); //0=default, -1=any camera, 1..99=your camera
	cv::Mat frame;

	bool cameraOn = capture.isOpened() && false;
	if (cameraOn) {
		if (!capture.read(frame))
			exit(3);
	} else {
		std::cerr << "No camera detected" << std::endl;
		frame = cv::imread("preview.png");
		if(frame.data == NULL)
			exit(3);
	}

	const unsigned int w = frame.cols;
	const unsigned int h = frame.rows;

	// resulting image after conversion is greyscale
	cv::Mat convertedFrame(h, w, CV_8UC1);

	// resulting image after sobelColor is color
	cv::Mat edgeFrame(h, w, CV_8UC1);

	cv::namedWindow("preview", 0);
	cv::namedWindow("converted", 0);

	while (((char)cv::waitKey(10)) <= -1) {
		if (cameraOn && !capture.read(frame))
			exit(3);

		greyFilter(frame.data, convertedFrame.data, w, h);
		sobelFilter(convertedFrame.data, convertedFrame.data, w, h,.5f);
		// show the result
		cv::imshow("preview", frame);
		cv::imshow("converted", convertedFrame);
	}

	cv::destroyWindow("preview");
	cv::destroyWindow("converted");

	return 0;
}
