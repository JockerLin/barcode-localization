#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd/linemod.hpp>
#include <opencv2/core.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

   Mat image, dst_image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    cv::linemod::ColorGradient();
    threshold(image, dst_image, 100, 255, cv::THRESH_BINARY);
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", dst_image);
    waitKey(0);

    return 0;
}