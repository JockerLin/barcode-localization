#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

int main()
{
    cv::Mat image;
    cout << "argc" << endl;
    image = imread("/home/pilcq/Pictures/messi.jpg");
    cout << image << endl;
    namedWindow("Show image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Show image", image);
    cv::waitKey(0);
    return 0;
}
