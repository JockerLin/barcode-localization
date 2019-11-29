#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "cv.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    cv::Mat image;
//    cv::waitKey();
//    cout << argc << endl;
//    if ( argc != 2 )
//    {
//        printf("usage123: DisplayImage.out <Image_Path>\n");
//    }
//    waitKey(500);
//
//    image = imread( argv[1], 1 );
//    if ( !image.data )
//    {
//        printf("No image data \n");
//        return -1;
//    }
    image = cv::imread("./test.png");
    namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    cv::waitKey(0);
    return 0;
}