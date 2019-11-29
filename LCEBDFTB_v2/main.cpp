#include "barcode_localization.h"


void nothing(int, void *) {
    ;
}

int main() {
    cv::Mat image;
    image = cv::imread("../barcode_32.bmp");
//    image = cv::imread("/home/pilcq/Pictures/tb_threshold.bmp");
//    cv::resize(image, image, cv::Size(640, 480));
    namedWindow("Paramemters", cv::WINDOW_NORMAL);
//    imshow("Paramemters", image);
//    cv::waitKey(0);
    std::vector<std::vector<cv::Point>> contours;

    int minLineLength = 100;//100
    // 候选线段分数阈值，分数来源于单个限定框内的最高分数，(得分根据线段与周围线段的关系决定 cpp390)
    int support_candidates_threshold = 2;//2

    // 与周围的delta条码数互相比较
    int delta = 8;//8
    // barcode整体长度与barcode内线段segment的比例 最大与最小
    int maxLengthToLineLengthRatio = 6;//6
    int minLengthToLineLengthRatio = 1;
    // 判断是否属于同一个barcode的阈值设置
    // barcode内segment的XY最大距离阈值
    int inSegmentXDistance = 65;//65
    int inSegmentYDistance = 65;//65

    if (DEBUGME) {
        cv::createTrackbar("minLineLength", "Paramemters", &minLineLength, 200, nothing);
        cv::createTrackbar("support_candidates_threshold", "Paramemters", &support_candidates_threshold, 90, nothing);
        cv::createTrackbar("delta", "Paramemters", &delta, 20, nothing);
        cv::createTrackbar("maxLengthToLineLengthRatio", "Paramemters", &maxLengthToLineLengthRatio, 90, nothing);
        cv::createTrackbar("minLengthToLineLengthRatio", "Paramemters", &minLengthToLineLengthRatio, 90, nothing);
        cv::createTrackbar("inSegmentXDistance", "Paramemters", &inSegmentXDistance, 90, nothing);
        cv::createTrackbar("inSegmentYDistance", "Paramemters", &inSegmentYDistance, 90, nothing);

        while (true) {
            if (cv::waitKey(500) == 27) {
                break;
            }
            minLineLength = cv::getTrackbarPos("minLineLength", "Paramemters");
            if (minLineLength < 0) {
                minLineLength = 0;
            }
            support_candidates_threshold = cv::getTrackbarPos("support_candidates_threshold", "Paramemters");
            if (support_candidates_threshold < 0) {
                support_candidates_threshold = 0;
            }
            delta = cv::getTrackbarPos("delta", "Paramemters");
            if (delta < 0) {
                delta = 0;
            }
            maxLengthToLineLengthRatio = cv::getTrackbarPos("maxLengthToLineLengthRatio", "Paramemters");
            if (maxLengthToLineLengthRatio < 0) {
                maxLengthToLineLengthRatio = 0;
            }
            minLengthToLineLengthRatio = cv::getTrackbarPos("minLengthToLineLengthRatio", "Paramemters");
            if (minLengthToLineLengthRatio < 0) {
                minLengthToLineLengthRatio = 0;
            }
            inSegmentXDistance = cv::getTrackbarPos("inSegmentXDistance", "Paramemters");
            if (inSegmentXDistance < 0) {
                inSegmentXDistance = 0;
            }
            inSegmentYDistance = cv::getTrackbarPos("inSegmentYDistance", "Paramemters");
            if (inSegmentYDistance < 0) {
                inSegmentYDistance = 0;
            }

            std::cout << delta << std::endl;
            contours = locateBarcode(
                    image,
                    minLineLength,
                    support_candidates_threshold,
                    delta,
                    maxLengthToLineLengthRatio,
                    minLengthToLineLengthRatio,
                    inSegmentXDistance,
                    inSegmentYDistance);
        }

    } else {
        contours = locateBarcode(
                image,
                minLineLength,
                support_candidates_threshold,
                delta,
                maxLengthToLineLengthRatio,
                minLengthToLineLengthRatio,
                inSegmentXDistance,
                inSegmentYDistance);
        cv::waitKey();
    }


    return 0;
}