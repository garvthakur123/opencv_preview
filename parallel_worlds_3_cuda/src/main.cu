#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CUDAInterface.hpp"
#include "GreyFilter.hpp"
#include "SobelFilter.hpp"
#include "EffectFilter.hpp"

GreyFilter greyFilter(3, 1);
SobelFilter sobelFilter(1, 1);
EffectFilter effectFilter(3, 1, 3);

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
		if (frame.data == NULL)
			exit(3);
	}

	const unsigned int w = frame.cols;
	const unsigned int h = frame.rows;

	// resulting image after conversion is greyscale
	cv::Mat convertedFrame(h, w, CV_8UC1);

	// resulting image after sobel edge detection is greyscale
	cv::Mat edgeFrame(h, w, CV_8UC1);

	// image after applying sobel-based effects is color, again
	cv::Mat effectFrame(h, w, CV_8UC3);

	cv::namedWindow("preview", 0);
	cv::namedWindow("converted", 0);
	cv::namedWindow("edge", 0);
	cv::namedWindow("effect", 0);

	while (((char)cv::waitKey(10)) <= -1) {
		if (cameraOn && !capture.read(frame))
			exit(3);

		greyFilter(frame.data, convertedFrame.data, w, h);
		sobelFilter(convertedFrame.data, edgeFrame.data, w, h, .5f);
		effectFilter(frame.data, edgeFrame.data, effectFrame.data, w, h, 90.0f);

		cv::imshow("preview", frame);
		cv::imshow("converted", convertedFrame);
		cv::imshow("edge", edgeFrame);
		cv::imshow("effect", effectFrame);
	}

	cv::destroyWindow("preview");
	cv::destroyWindow("converted");
	cv::destroyWindow("edge");
	cv::destroyWindow("effect");

	return 0;
}
