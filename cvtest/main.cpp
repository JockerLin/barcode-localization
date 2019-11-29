//#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/rgbd/linemod.hpp>


//using namespace std;
using namespace cv;

void printHello()
{
    std::cout << "hello" << std::endl;
}

int main()
{
    cv::Mat image;
    image = cv::imread("/home/pilcq/Pictures/messi.jpg");
//    cout << image << endl;
    namedWindow("Show image", cv::WINDOW_AUTOSIZE);
    imshow("Show image", image);
    cv::imwrite("debug-line-segments.jpg", image);
    cv::waitKey(0);
    printHello();
    return 0;
}
