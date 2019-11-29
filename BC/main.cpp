#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

//using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;
    cv::Mat image;
    image = cv::imread("/home/pilcq/personal/PROJECT_CODE!/c_project/BC/messi.jpg");
    namedWindow("Show image", cv::WINDOW_AUTOSIZE);
    imshow("Show image", image);
    cv::waitKey(0);
    return 0;
}