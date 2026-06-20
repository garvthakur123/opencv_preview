#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the image
    cv::Mat image = cv::imread("preview.png");

    // Check if image loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not load image 'preview.png'" << std::endl;
        return -1;
    }

    // Show the image in a window
    cv::imshow("preview", image);

    // Wait for a key press
    cv::waitKey(0);

    // Save the image
    if (!cv::imwrite("preview_new.png", image)) {
        std::cerr << "Error: Could not save image 'preview_new.png'" << std::endl;
        return -1;
    }

    std::cout << "Image saved as preview_new.png" << std::endl;

    return 0;
}
