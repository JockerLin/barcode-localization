#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>


int main(int argc, char** argv) {
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if(!image.data) {
        std::cout << "Error: the image wasn't correctly loaded." << std::endl;
        return -1;
    }

    // We iterate over all pixels of the image
    for(int r = 0; r < image.rows; r++) {
        // We obtain a pointer to the beginning of row r
        cv::Vec3b* ptr = image.ptr<cv::Vec3b>(r);

        for(int c = 0; c < image.cols; c++) {
            // We invert the blue and red values of the pixel
            ptr[c] = cv::Vec3b(ptr[c][2], ptr[c][1], ptr[c][0]);
        }
    }

    cv::imshow("Inverted Image", image);
    cv::waitKey();

    return 0;
}