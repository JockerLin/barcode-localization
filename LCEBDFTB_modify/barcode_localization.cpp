#include "barcode_localization.h"

std::vector<std::vector<cv::Point>> locateBarcode(cv::Mat image_color,
                                                  int minLineLength,//最小检测直线的长度，小于该长度则忽略
                                                  int support_candidates_threshold,//检测contour的阈值，小于该值忽略
                                                  int delta,//条码竖线段的比较数？？？
                                                  int maxLengthToLineLengthRatio,//忽略轮廓高值 比例
                                                  int minLengthToLineLengthRatio,//忽略轮廓低值 比例
                                                  int inSegmentXDistance,//x方向线段与轮廓的距离
                                                  int inSegmentYDistance)//x方向线段与轮廓的距离
{

//----------------------------part 1 find bounding box of line segment-----------------------------------------------------------------
    cv::Mat image_greyscale;
    cv::cvtColor(image_color, image_greyscale, CV_BGR2GRAY);

    // Create LSDDetector
    cv::line_descriptor::LSDDetector LSD;

    // Create keylines vector
    std::vector<cv::line_descriptor::KeyLine> keylines;
    // Detect lines with the LSD  keylines为LSD检测到的结果
    auto start = std::chrono::steady_clock::now();
    auto total_start = start;
    LSD.detect(image_greyscale, keylines, 2, 1);
    // 两个参数意义 金字塔生成中使用的比例因子
    auto end = std::chrono::steady_clock::now();
    std::cout << "LSD: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // Draw the detected lines on the image
    cv::Mat image_lines;
    cv::Mat image_key_lines;
    image_color.copyTo(image_lines);
    image_color.copyTo(image_key_lines);

    for (int i=0; i<keylines.size(); i++){
        if (keylines[i].lineLength > minLineLength){
            cv::line(image_key_lines, keylines[i].getStartPoint(), keylines[i].getEndPoint(), cv::Scalar(0, 0, 255), 3);
        }
    }
    if (! DEBUGME){
        cv::imwrite("lsd_lines.jpg", image_key_lines);
    }
    cv::namedWindow("image_key_lines", cv::WINDOW_NORMAL);
    cv::imshow("image_key_lines", image_key_lines);

    std::cout << "Number of lines detected: " << keylines.size() << std::endl;

    start = std::chrono::steady_clock::now();
    // 对每个满足条件的key line建立contours
    std::vector<std::vector<cv::Point>> contours_lineSegments = getLineSegmentsContours(keylines, image_lines,
                                                                                        minLineLength);
    end = std::chrono::steady_clock::now();
    std::cout << "Creating bounding boxes: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
              << std::endl;

    cv::drawContours(image_lines, contours_lineSegments, -1, cv::Scalar(255, 0, 0));

    std::cout << "Number of contours_lineSegments: " << contours_lineSegments.size() << ",color:blue" << std::endl;

    if (! DEBUGME){
        cv::imwrite("debug-line-segments.jpg", image_lines);
    }
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::imshow("Image", image_lines);


//----------------------------------part 2 选择候选线段--------------------------------------------------------------------------------
    // Find for every bounding box the containing segments
    std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> keylinesInContours(
            contours_lineSegments.size());
    int contours_lineSegments_size = contours_lineSegments.size();

    start = std::chrono::steady_clock::now();
    // 因为bounding box 几乎是每个线段的四倍大小，所以在bounding box 里选择包含的segments
    findContainingSegments(keylinesInContours, keylines, contours_lineSegments, contours_lineSegments_size);
    end = std::chrono::steady_clock::now();
    std::cout << "Find segments in bounding boxes: " << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms" << std::endl;

    std::vector<std::vector<int>> support_scores(keylinesInContours.size());

    // Calculate support score of every segment for every bounding box
    // 计算每个边界框的每个段的支持分数
    int keylinesInContours_size = keylinesInContours.size();
    start = std::chrono::steady_clock::now();
    calculateSupportScores(keylinesInContours, support_scores, keylinesInContours_size);
    //若满足条件的线段对 分数会+1， support_scores储存着每个contours的每条线的分数
    end = std::chrono::steady_clock::now();
    std::cout << "Calculate support scores: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
              << std::endl;

    // Select s_cand segment candidate 选择候选线段
    start = std::chrono::steady_clock::now();

    std::vector<int> support_candidates(keylinesInContours_size);
    std::vector<int> support_candidates_pos(keylinesInContours_size);
    cv::Mat image_candidates;
    image_color.copyTo(image_candidates);

    // 选择候选线段 更新support_candidates_pos与support_candidates
    // 获取所有contour内的最高分数线段的分数，及其分数
    selectSCand(support_scores,
                support_candidates,
                support_candidates_pos,
                keylinesInContours,
                keylinesInContours_size,//生成的太多相似的 需要过滤
                image_candidates,
                support_candidates_threshold);
    end = std::chrono::steady_clock::now();
    std::cout << "Select s_cand: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
              << std::endl;

//----------------------------------part 3 Calculate bounding boxes-------------------------------------------------------------------
// ready to understand!!!!
// Create vectors of intensities
    // 创建向量的强度
    start = std::chrono::steady_clock::now();

    std::vector<bool> deletedContours(keylinesInContours_size);
    for (bool &&dc : deletedContours) {
        dc = false;
    }

    std::vector<std::vector<cv::Point>> perpendicularLineStartEndPoints(keylinesInContours_size,
                                                                        std::vector<cv::Point>(2));
    std::vector<std::vector<std::vector<uchar>>> intensities(keylinesInContours_size,
                                                             std::vector<std::vector<uchar>>(5));
    std::vector<std::vector<int>> startStopIntensitiesPosition(keylinesInContours_size, std::vector<int>(2));

    int intensities_size = intensities.size();
    int image_cols = image_greyscale.cols;
    int image_rows = image_greyscale.rows;
    std::vector<cv::line_descriptor::KeyLine> perpendicularLines;
    // 计算每个contour内 每条分割线的每个点的 像素强度(gray pixel value) ok
    createVectorsOfIntensities(support_candidates,
                               support_candidates_pos,
                               keylinesInContours,
                               startStopIntensitiesPosition,
                               perpendicularLineStartEndPoints,
                               intensities,
                               image_greyscale,
                               image_cols,
                               image_rows,
                               intensities_size,
                               support_candidates_threshold,
                               deletedContours,
                               perpendicularLines);

    //对图像与barcode平分线的处理
    myProcess(image_color, perpendicularLines);

    end = std::chrono::steady_clock::now();
    std::cout << "Compute intensities: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
              << std::endl;

    std::cout << "Total time until my process: " << std::chrono::duration<double, std::milli>(end - total_start).count() << " ms"
              << std::endl;

    // Compute phis
    start = std::chrono::steady_clock::now();
    std::vector<std::vector<std::vector<int>>> phis(keylinesInContours_size, std::vector<std::vector<int>>(6));
    std::vector<int> start_barcode_pos(keylinesInContours_size);
    std::vector<int> end_barcode_pos(keylinesInContours_size);

    // 计算barcode的开始结束位置
    computePhis(delta,
                intensities,
                intensities_size,
                phis,
                startStopIntensitiesPosition,
                start_barcode_pos,
                end_barcode_pos,
                deletedContours,
                image_color);
    end = std::chrono::steady_clock::now();
    std::cout << "Compute phis: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
              << std::endl;

    //debug 绘制 start-end barcode
//    std::cout<<"start_barcode_pos:"<<start_barcode_pos.size()<<std::endl;
//    for(int index=0; index<start_barcode_pos.size(); index++){
//        cv::line(image_color, start_barcode_pos[index], end_barcode_pos[index], cv::Scalar(255, 150, 50));
//    }
//    cv::namedWindow("barcode_pos");
//    cv::imshow("barcode_pos", image_color);
//    cv::waitKey();

    // For debugging
    // Select a good line segment example
    /*
    int index = 0;
    int nmb = 0;
    bool finish = false;
    for(unsigned int i = 0; (i < intensities.size()) && (!finish); i++) {
        if(0 < intensities[i][3].size()) {
            nmb++;
            //if(250 < nmb) {
            if(290 < nmb) {
                finish = true;
                index = i;
            }
        }
    }
    index = 0;
    std::cout << "index = " << index << ", size() = " << intensities[index][2].size() << std::endl;
    */
    int index = 0;

    // Calculate bounding boxes
    start = std::chrono::steady_clock::now();
    std::vector<std::vector<cv::Point>> contours_barcodes(keylinesInContours_size, std::vector<cv::Point>(4));
    //计算barcode的限定框
    calculateBoundingBoxes(keylinesInContours_size,
                           start_barcode_pos,
                           end_barcode_pos,
                           keylines,
                           contours_barcodes,
                           perpendicularLineStartEndPoints,
                           image_candidates,
                           deletedContours,
                           index,
                           maxLengthToLineLengthRatio,
                           minLengthToLineLengthRatio);

    end = std::chrono::steady_clock::now();
    std::cout << "Calculated bounding boxes: " << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms" << std::endl;

//    if (contours_barcodes.size()>0)
//        cv::drawContours(image_candidates, contours_barcodes, -1, cv::Scalar(255, 255, 0), 1);
//        std::cout<<contours_barcodes.size()<<std::endl;
//        cv::imshow("image_candidates", image_candidates);
//        cv::waitKey();
    std::cout << "before filter: " << contours_barcodes.size() << std::endl;

    // Filtering bounding boxes
    start = std::chrono::steady_clock::now();

    filterContours(keylinesInContours_size,
                   deletedContours,
                   start_barcode_pos,
                   end_barcode_pos,
                   keylines,
                   support_scores,
                   contours_barcodes,
                   inSegmentXDistance,
                   inSegmentYDistance);

    std::cout << "after filter: " << contours_barcodes.size() << std::endl;

//    if (contours_barcodes.size()>0)
//        cv::drawContours(image_candidates, contours_barcodes, -1, cv::Scalar(255, 255, 0), 1);

    for (int i = 0; i < keylinesInContours_size; i++) {
        if (false == deletedContours[i]) {
            cv::putText(image_candidates, std::to_string(i), contours_barcodes[i][0], cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 0, 255));
        }
    }
//    cv::imshow("image_candidates", image_candidates);
//    cv::waitKey();

//    if (! DEBUGME){
//        cv::imwrite("debug_candidate_segments.jpg", image_candidates);
//    }
//    cv::namedWindow("image_candidates", cv::WINDOW_NORMAL);
//    cv::imshow("image_candidates", image_candidates);
    end = std::chrono::steady_clock::now();
    std::cout << "Filtering bounding boxes: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
              << std::endl;

    // barcode decoding with ZBar

    start = std::chrono::steady_clock::now();

    cv::Mat image_barcodes;
    image_color.copyTo(image_barcodes);

    std::vector<std::string> barcodes = decodeBarcode(keylinesInContours_size, deletedContours, contours_barcodes,
                                                      image_greyscale, image_barcodes);

    end = std::chrono::steady_clock::now();
    std::cout << "Barcode decoding: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
              << std::endl;

    std::cout << "Total time: " << std::chrono::duration<double, std::milli>(end - total_start).count() << " ms"
              << std::endl;


    return (contours_barcodes);
}

// 对每个满足条件的key line建立contours
std::vector<std::vector<cv::Point>> getLineSegmentsContours(std::vector<cv::line_descriptor::KeyLine> &keylines,
                                                            cv::Mat &image_lines,
                                                            int minLineLength) {
    std::vector<std::vector<cv::Point>> contours;
    // Go throu all the keylines which were detected by the LSDetector
    for (auto it = keylines.begin(); it != keylines.end();) {
        auto kl = *it;
        // Only process lines which are greater than minLIneLength
        if (minLineLength < kl.lineLength) {
            // For debug
            //cv::line(image_lines, kl.getStartPoint(), kl.getEndPoint(), cv::Scalar(255, 0, 0));

            // Define start and end point so that the line points upwards.
            // kl.angle 为line相对正向x轴的斜率
            float linelength = kl.lineLength;
            float angle = kl.angle;
            float cos_angle = std::abs(std::cos(angle));
            float sin_angle = std::abs(std::sin(angle));
            float start_x;
            float start_y;
            float end_x;
            float end_y;
            // 取y较大的为开始点
            if (kl.startPointY > kl.endPointY) {
                start_x = kl.startPointX;
                start_y = kl.startPointY;
                end_x = kl.endPointX;
                end_y = kl.endPointY;
            } else {
                start_x = kl.endPointX;
                start_y = kl.endPointY;
                end_x = kl.startPointX;
                end_y = kl.startPointY;
            }

            // Create contour which is 4 times the line length in x-direction.
            // 对每个key line 建立contours ,为什么以这样的尺寸建立contour????????????1
            float temp_1 = 2 * linelength * sin_angle;
            float temp_2 = 5.0 * linelength * cos_angle;

            std::vector<cv::Point> contour(5);
            contour[0] = (cv::Point2f(start_x - temp_1, start_y + temp_2));
            contour[1] = (cv::Point2f(start_x + temp_1, start_y + temp_2));
            contour[2] = (cv::Point2f(end_x + temp_1, end_y - temp_2));
            contour[3] = (cv::Point2f(end_x - temp_1, end_y - temp_2));
            contour[4] = (cv::Point2f(start_x - temp_1, start_y + temp_2));
            contours.push_back(contour);

            ++it;
        } else { // Erase the keyline if it is too short.
            keylines.erase(it);
        }
    }

    return (contours);
}

void findContainingSegments(std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
                            std::vector<cv::line_descriptor::KeyLine> keylines,
                            std::vector<std::vector<cv::Point>> contours,
                            int contours_size) {
    //keylines 为LSD检测的segment contours为长度大于minlength的keylines的contours
    //#pragma omp parallel for
    // Process every contour 处理每个contours
    for (int i = 0; i < contours_size; i++) {
        register float px = keylines[i].pt.x;
        register float py = keylines[i].pt.y;
        register float ll_2 = keylines[i].lineLength * 2;

        register int keylines_size = keylines.size();
        //#pragma omp parallel for
        // Process every keyline
        for (int j = 0; j < keylines_size; j++) {
            // Add the keyline with the same number as the contour as its the contour based on this line.
            // 添加与contour相同的关键线，使其contour基于当前线段。
            // i==j 的含义 Contours的index==keylines的index??????????
            if (i == j) {
                keylinesInContours[i].push_back(std::make_shared<cv::line_descriptor::KeyLine>(keylines[j]));
            } else {
                register cv::line_descriptor::KeyLine kl_j = keylines[j];
                // Consider only keylines which are closer as ll_2 in x- and y-direction
                if (std::abs(kl_j.pt.x - px) < ll_2) {
                    if (std::abs(kl_j.pt.y - py) < ll_2) {
                        // Check if segment is in contour
                        if ((0 < cv::pointPolygonTest(contours[i], kl_j.getStartPoint(), false)) &&
                            (0 < cv::pointPolygonTest(contours[i], kl_j.getEndPoint(), false))) {
                            keylinesInContours[i].push_back(
                                    std::make_shared<cv::line_descriptor::KeyLine>(keylines[j]));
                        }
                    }
                }
            }
        }
    }
}

void calculateSupportScores(std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
                            std::vector<std::vector<int>> &support_scores,
                            int keylinesInContours_size) {
    //#pragma omp parallel for
    // Process every contour
    for (int i = 0; i < keylinesInContours_size; i++) {
        int keylinesInContours_i_size = keylinesInContours[i].size();
        support_scores[i] = std::vector<int>(keylinesInContours[i].size());
        //#pragma omp parallel for
        // Initialize support_scores with 0.
        // 初始化每个contours的每条线初始分数为0 尺寸为先前计算的len
        for (int j = 0; j < keylinesInContours_i_size; ++j) {
            support_scores[i][j] = 0;
        }
    }

    register float diff_length;
    register float diff_angle;
    register float diff_norm_pt;
    //#pragma omp parallel for
    // Check in every contour every possible keyline pair.
    // 检查每个限定框的keyline线对
    for (int i = 0; i < keylinesInContours_size; i++) {
        int keylinesInContours_i_size = keylinesInContours[i].size();//第i个contours内所有线的数目
        //#pragma omp parallel for
        for (int j = 0; j < keylinesInContours_i_size; j++) {
            register std::shared_ptr<cv::line_descriptor::KeyLine> kl_j = keylinesInContours[i][j];//寄存器变量 第i个contours的第j条线

            for (int k = j + 1; k < keylinesInContours_i_size; k++) {
                register std::shared_ptr<cv::line_descriptor::KeyLine> kl_k = keylinesInContours[i][k];//第i个contours的第k条线

                diff_length = std::abs(kl_j->lineLength - kl_k->lineLength);
                // 自定义对每个kl的过滤
                // Check length difference.
                if ((diff_length) < 4.0) {
                    diff_angle = std::abs(kl_j->angle - kl_k->angle);

                    // Check angle difference. 35.5°
                    if ((diff_angle) < 0.26) {
                        diff_norm_pt = cv::norm(kl_j->pt - kl_k->pt);// 范数

                        // Check position difference
                        if ((diff_norm_pt) < 300.0) {
                            // Increase the support scores of the pair if all threshold are fine.
//                            std::cout<<"diff_length: "<<diff_length<<"diff_angle: "<<diff_angle<<"diff_norm_pt: "<<diff_norm_pt<<std::endl;
                            support_scores[i][j] += 1;
                            support_scores[i][k] += 1;
                        }
                    }
                }
            }
        }
    }
}

void selectSCand(std::vector<std::vector<int>> &support_scores,
                 std::vector<int> &support_candidates,
                 std::vector<int> &support_candidates_pos,
                 std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
                 int keylinesInContours_size,
                 cv::Mat &image_candidates,
                 int support_candidates_threshold) {

    //#pragma omp parallel for
    int count = 0;
    //历遍每个contours
    for (int i = 0; i < keylinesInContours_size; i++) {
        // Get position of the maximum element.
        // support_scores[i]包含该contours内所有线段的置信度(support scores)分数值
        // 获取第i个contour内的最高分数线段的位置
        support_candidates_pos[i] = std::distance(support_scores[i].begin(),
                                                  std::max_element(support_scores[i].begin(),
                                                                   support_scores[i].end()));

        // Get support value of the maximum element.
        // 获取第i个contour内的最高分数线段的分数
        support_candidates[i] = support_scores[i][std::distance(support_scores[i].begin(),
                                                                std::max_element(support_scores[i].begin(),
                                                                                 support_scores[i].end()))];
//        support_candidates[i] = support_scores[i][support_candidates_pos[i]]

        // For debug
        if (support_candidates_threshold < support_candidates[i]) {
            //若满足分数阈值
            count++;
//            std::cout<<count++<<" data:"<<support_candidates_threshold<<"<support_candidates[i]="<<support_candidates[i]<<std::endl;
            std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];
            cv::line(image_candidates, kl->getStartPoint(), kl->getEndPoint(), cv::Scalar(0, 255, 0), 5);
        }
    }
    std::cout << "number of select segment candidate:" << count << std::endl;
    cv::namedWindow("image_candidates", cv::WINDOW_NORMAL);
    cv::imshow("image_candidates", image_candidates);
    if (! DEBUGME){
        cv::imwrite("image_candidates_segments.jpg", image_candidates);
    }

//    cv::waitKey(0);
}

//计算每个contour内 每条分割线的每个点的 像素强度(gray pixel value)
void createVectorsOfIntensities(std::vector<int> &support_candidates,
                                std::vector<int> &support_candidates_pos,
                                std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
                                std::vector<std::vector<int>> &startStopIntensitiesPosition,//包含强度矢量的起始和停止位置的矢量。
                                std::vector<std::vector<cv::Point>> &perpendicularLineStartEndPoints,//垂线的起点和终点。
                                std::vector<std::vector<std::vector<uchar>>> &intensities,//包含每条等高线的五条平行线point的强度的向量。
                                cv::Mat &image_greyscale,
                                int image_cols,
                                int image_rows,
                                int intensities_size,
                                int support_candidates_threshold,//在该阈值下等值线被忽略。
                                std::vector<bool> &deletedContours,
                                std::vector<cv::line_descriptor::KeyLine> &perpendicularLines) {
    float angle;
    float kl_pt_x;
    float kl_pt_y;
    float temp_0;
    float temp_1;
    float temp_start_y;
    int temp_start_x;
    int temp_end_x;
    int temp_start_mock_x;
    int temp_start_mock_y;
    int temp_end_mock_x;
    int temp_end_mock_y;
    std::vector<cv::Point> pt1s(6);
    float temp_3;
    float temp_end_y;
    std::vector<cv::Point> pt2s(6);
    int pt_size;
    int start;
    int end;
    int lineIterators_size_2;
    int lineIterators_5_count;


    // Process every contour.
    // intensities_size 相似的过多了 需要极大值
//    std::cout<<"intensities_size is :"<<intensities_size<<std::endl;
//    std::cout<<"intensities_size is :"<<intensities_size<<std::endl;
//    std::cout<<"intensities_size is :"<<intensities_size<<std::endl;

    for (int i = 0; i < intensities_size; i++) {
        // Process only candidates above the support threshold
        if (support_candidates_threshold < support_candidates[i]) {
            std::shared_ptr<cv::line_descriptor::KeyLine> kl = keylinesInContours[i][support_candidates_pos[i]];
            // 角度:直线与正向x轴所成角度
            angle = kl->angle;
            //std::cout << "angle = " << 180*angle/M_PI << std::endl;
            // Decrease the angel by 90 degree if greater than 90 degree to remove ambiguity. 减去90度消除歧义
            if (M_PI_2 < angle) {
                angle -= M_PI_2;
            }
            cv::Mat image_debug_temp;
            image_greyscale.copyTo(image_debug_temp);
            cv::cvtColor(image_greyscale, image_debug_temp, CV_GRAY2BGR);

            if (DEBUGDETAIL) {
                cv::line(image_debug_temp, kl->getStartPoint(), kl->getEndPoint(), cv::Scalar(0, 255, 0), 5);
                std::cout << "pos kl :(" << kl->pt.x << "," << kl->pt.y << ")" << std::endl;
                std::cout << "kl angle:"<<kl->angle<<std::endl;
                cv::namedWindow("image_temp 578", cv::WINDOW_NORMAL);
                cv::imshow("image_temp 578", image_debug_temp);
                cv::waitKey();
            }


            kl_pt_y = kl->pt.y;
            kl_pt_x = kl->pt.x;

            // Handel different angel cases.
            // 为什么要用九十度来减，变角度值
            if (M_PI_4 > std::abs(angle)) {
                if (0 < angle) {
                    angle = (M_PI_2 - angle);
                } else {
                    angle = -(M_PI_2 - std::abs(angle));
                }

                // Calculate start and end points
                temp_1 = 2000 * std::sin(angle);
                temp_start_x = kl_pt_x - temp_1;
                temp_start_y = kl_pt_y - temp_1 * (1 / std::tan(angle));
                temp_end_x = kl_pt_x + temp_1;
                temp_end_y = kl_pt_y + temp_1 * (1 / std::tan(angle));

                temp_start_mock_x = kl_pt_x - kl_pt_y * std::tan(angle);
                temp_start_mock_y = 0;
                temp_end_mock_x = kl_pt_x + (image_rows - kl_pt_y) * std::tan(angle);
                temp_end_mock_y = image_rows;
            } else {
                // Handel different angel cases.
                if (0 < angle) {
                    angle = (M_PI_2 - angle);
                } else {
                    angle = -(M_PI_2 - std::abs(angle));
                }

                // Calculate start and end points
                temp_1 = 2000 * std::cos(angle);
                temp_start_x = kl_pt_x - temp_1;
                temp_start_y = kl_pt_y + temp_1 * std::tan(angle);
                temp_end_x = kl_pt_x + temp_1;
                temp_end_y = kl_pt_y - temp_1 * std::tan(angle);

                temp_start_mock_x = 0;
                temp_start_mock_y = kl_pt_y + kl_pt_x * std::tan(angle);
                temp_end_mock_x = image_cols;
                temp_end_mock_y = kl_pt_y - (image_cols - kl_pt_x) * std::tan(angle);
            }

//            std::cout << "temp_start_x:" << temp_start_x << std::endl;
//            std::cout << "temp_start_mock_x:" << temp_start_mock_x << std::endl;
//            if (temp_start_x<temp_start_mock_x){
//                std::cout<<"<"<<std::endl;
//                temp_start_x = temp_start_mock_x;
//            }
//
//            if (temp_end_x>temp_end_mock_x){
//                temp_end_x = temp_end_mock_x;
//            }

            startStopIntensitiesPosition[i][0] = temp_start_x;
            startStopIntensitiesPosition[i][1] = temp_end_x;

            // Create points for the lineIterators.
            pt1s[0] = (cv::Point(temp_start_x, temp_start_y - 16));
            pt1s[1] = (cv::Point(temp_start_x, temp_start_y - 8));
            pt1s[2] = (cv::Point(temp_start_x, temp_start_y));
            pt1s[3] = (cv::Point(temp_start_x, temp_start_y + 8));
            pt1s[4] = (cv::Point(temp_start_x, temp_start_y + 16));
            pt1s[5] = cv::Point(temp_start_mock_x, temp_start_mock_y);

            pt2s[0] = (cv::Point(temp_end_x, temp_end_y - 16));
            pt2s[1] = (cv::Point(temp_end_x, temp_end_y - 8));
            pt2s[2] = (cv::Point(temp_end_x, temp_end_y));
            pt2s[3] = (cv::Point(temp_end_x, temp_end_y + 8));
            pt2s[4] = (cv::Point(temp_end_x, temp_end_y + 16));
            pt2s[5] = cv::Point(temp_end_mock_x, temp_end_mock_y);

            temp_0 = kl_pt_y + kl_pt_x * std::tan(angle);
            temp_3 = kl_pt_y - (image_cols - kl_pt_x) * std::tan(angle);

            //垂线的起点与终点
            perpendicularLineStartEndPoints[i][0] = cv::Point(temp_start_mock_x, temp_start_mock_y);
            perpendicularLineStartEndPoints[i][1] = cv::Point(temp_end_mock_x, temp_end_mock_y);

            // Create lineIterators
            // 创建线段迭代器
            pt_size = pt1s.size();
//            std::cout<<"pt_size:"<<pt_size<<std::endl;

            std::vector<cv::LineIterator> lineIterators;
            for (int j = 0; j < pt_size; j++) {
                lineIterators.push_back(cv::LineIterator(image_greyscale, pt1s[j], pt2s[j], 8, true));
//                std::cout<<"lineIterators[j].count"<<lineIterators[j].count<<std::endl;
//                std::cout<<pt1s[j].x<<","<<pt1s[j].y<<std::endl;
//                std::cout<<pt2s[j].x<<","<<pt2s[j].y<<std::endl;

                // lineIterators[j].count 产生640个点 因为图像的宽度为640?
//                std::cout<<"j:"<<j<<std::endl;
//                cv::line(image_temp, pt1s[j], pt2s[j], cv::Scalar(0, 255, 0), 1);
            }

            float line_angle;
            cv::line_descriptor::KeyLine line;
//            line_angle = float(pt2s[2].y-pt1s[2].y)/float(pt2s[2].x-pt1s[2].x);
            line.angle = float(pt2s[2].y-pt1s[2].y)/float(pt2s[2].x-pt1s[2].x);
            line.startPointX = pt1s[2].x;
            line.startPointY = pt1s[2].y;
            line.endPointX = pt2s[2].x;
            line.endPointY = pt2s[2].y;
            perpendicularLines.push_back(line);

            if (DEBUGDETAIL) {
                // debug linelineline to check perpendicular line
                cv::line(image_debug_temp, line.getStartPoint(), line.getEndPoint(), cv::Scalar(0, 255, 0), 2);
                std::cout << "angle :" << line.angle << std::endl;
                std::cout << "pos1 :(" << pt1s[2].x << "," << pt1s[2].y << ")" << std::endl;
                std::cout << "pos2 :(" << pt2s[2].x << "," << pt2s[2].y << ")" << std::endl;
                cv::namedWindow("image_temp 578", cv::WINDOW_NORMAL);
                cv::imshow("image_temp 578", image_debug_temp);
                cv::waitKey();
            }
            //此时已经找到barcode的中垂直线，需要裁剪直线为线段，确定barcode的起始结束点

            int temp_value = 0;
            int start_value = 0;
            int end_value = 0;
            // Find start and end positon of the shortened lineIterators with the help of the mock lineIterator (lineIterators[5])
            // 找到开始与结束线段描述子的位置
            // lineIterators[5]为参考线段 起始点x=0，结束点x=图像宽度
            if (lineIterators[5].count > 0) {
                for (start = 0; (temp_start_x > lineIterators[5].pos().x) && (start < lineIterators[5].count);
                     ++lineIterators[5], start++) {
                    temp_value = start;
                    start_value = start;
                }
            }
            if (lineIterators[5].count > 0) {
                for (end = temp_value; (temp_end_x > lineIterators[5].pos().x) && (end < lineIterators[5].count);
                     ++lineIterators[5], end++) {
                    end_value = end;
                }
            }

            lineIterators_size_2 = lineIterators.size() - 1;
            for (int j = 0; j < lineIterators_size_2; j++) {
                lineIterators_5_count = lineIterators[5].count;//640个点 图像宽度
                //第i个contour的第j条平分线
                intensities[i][j] = std::vector<uchar>(lineIterators_5_count);

                // Initialize intensity vector with 0.
                for (uchar &intensity : intensities[i][j]) {
                    intensity = 0;
                }

                // Get intensities of the shortened line.
                if (lineIterators[j].count > 0) {
                    for (int k = start_value; k < end_value; k++, ++lineIterators[j]) {
                        intensities[i][j][k] = image_greyscale.at<uchar>(lineIterators[j].pos());
                        //i个contour j条分割线上的 k个点 其强度为 intensities[i][j][k]
                    }
                }
            }
        } else { // If below the threshold, disable contour.
            deletedContours[i] = true;
        }
    }
}


void myProcess(cv::Mat &image_color, std::vector<cv::line_descriptor::KeyLine> &perpendicularLines) {
    cv::line_descriptor::KeyLine line;
    float angle_line;
    //std::tan(angle)
    float all_line_angle=0.0;
    float average_angle;
    cv::Mat image_gray;
    for (int i = 0; i < perpendicularLines.size(); i++) {
        cv::Mat image_color_temp;
        image_color.copyTo(image_color_temp);
        line = perpendicularLines[i];

        if (DEBUGMYPRO) {
            // debug linelineline to check perpendicular line
            cv::line(image_color_temp, line.getStartPoint(), line.getEndPoint(), cv::Scalar(0, 255, 0), 2);
            std::cout << "angle :" << line.angle << std::endl;
            cv::namedWindow("image_color_temp", cv::WINDOW_NORMAL);
            cv::imshow("image_color_temp", image_color_temp);
            cv::waitKey();
        }
        //待加入一些滤波 暂时是取平均的方法
        all_line_angle += line.angle;

        //对每条平分线求直方图

//        std::cout << "line.angle :" << line.angle << std::endl;
//        std::cout << "all_line_angle :" << all_line_angle << std::endl;
    }
    average_angle = all_line_angle/float(perpendicularLines.size());
    std::cout << "average_angle :" << average_angle << std::endl;
    //得到线段的平均角度 将图片旋转识别



}


// 计算barcode的开始结束位置
void computePhis(int delta,
                 std::vector<std::vector<std::vector<uchar>>> &intensities,
                 int intensities_size,
                 std::vector<std::vector<std::vector<int>>> &phis,
                 std::vector<std::vector<int>> &startStopIntensitiesPosition,
                 std::vector<int> &start_barcode_pos,
                 std::vector<int> &end_barcode_pos,
                 std::vector<bool> &deletedContours,
                 cv::Mat &image_color) {
    int phis_i_5_k;
    //#pragma omp parallel for
    // Process every contour
    std::cout << "all contours intensities_size: " << intensities_size << std::endl;
    std::cout << "deletedContours size: " << deletedContours.size() << std::endl;
    std::cout << "intensities[0].size() : " << intensities[0].size() << std::endl;
    for (int i = 0; i < intensities_size; i++) {
        int intensities_i_size = intensities[i].size();//5
        int startStopIntensitiesPosition_i_0 = startStopIntensitiesPosition[i][0];
        int startStopIntensitiesPosition_i_1 = startStopIntensitiesPosition[i][1];
        //平分线的起始点x与结束点x
//        std::cout<<"startStopIntensitiesPosition_i_0: "<<startStopIntensitiesPosition_i_0<<std::endl;
//        std::cout<<"startStopIntensitiesPosition_i_1: "<<startStopIntensitiesPosition_i_1<<std::endl;

        // Only consider still active candidates.
        if (false == deletedContours[i]) {
            //#pragma omp parallel for
            // Process all the five parallel lines.
            for (int j = 0; j < intensities_i_size; j++) {
                //对平分线的处理
                int intensities_i_j_size = intensities[i][j].size();//点的个数
                phis[i][j] = std::vector<int>(intensities_i_j_size);
                //#pragma omp parallel for
                for (int k = 0; k < intensities_i_j_size; k++) {
                    // Only calculate phi values for areas, wher phi is not 0.
                    //对平行线上点的处理,k为点的index(0, 640)
                    //startStopIntensitiesPosition_i_0 为平分线的start x 坐标 0
                    //startStopIntensitiesPosition_i_1 为平分线的end x 坐标 640
//                    std::cout << "startStopIntensitiesPosition_i_0:" << startStopIntensitiesPosition_i_0 << std::endl;
//                    std::cout << "startStopIntensitiesPosition_i_1:" << startStopIntensitiesPosition_i_1 << std::endl;
                    if (startStopIntensitiesPosition_i_0 - delta < k) {
                        if (startStopIntensitiesPosition_i_1 + delta > k) {
                            // Determine start and stop position.
                            int phi_1 = 0;
                            int phi_2 = 0;

                            int start_1 = k - delta - 1;
                            int end_1 = k;

                            if (startStopIntensitiesPosition_i_1 < k) {
                                if (start_1 < startStopIntensitiesPosition_i_1) {
                                    end_1 = startStopIntensitiesPosition_i_1;
                                }
                            }

                            if (end_1 > startStopIntensitiesPosition_i_0) {
                                if (startStopIntensitiesPosition_i_0 > start_1) {
                                    start_1 = startStopIntensitiesPosition_i_0;
                                }
                            }

                            // Calculate first part of phi.
                            //#pragma omp parallel for
                            for (int l = start_1; l < end_1; l++) {
                                phi_1 += std::abs(intensities[i][j][l + 1] - intensities[i][j][l]);
                            }

                            // Determine start and stop position.
                            int start_2 = k;
                            int end_2 = intensities_i_j_size;
                            if (startStopIntensitiesPosition_i_0 > start_2) {
                                if (end_1 > startStopIntensitiesPosition_i_0) {
                                    start_2 = startStopIntensitiesPosition_i_0;
                                }
                            }

                            if (intensities_i_j_size > (k + delta + 1)) {
                                end_2 = k + delta + 1;
                            }

                            if (startStopIntensitiesPosition_i_1 < end_2) {
                                if (start_2 < startStopIntensitiesPosition_i_1) {
                                    end_2 = startStopIntensitiesPosition_i_1;
                                }
                            }

                            // Calculate second part of phi.
                            //#pragma omp parallel for
                            for (int l = start_2; l < end_2; l++) {
                                phi_2 += std::abs(intensities[i][j][l] - intensities[i][j][l + 1]);
                            }

                            phis[i][j][k] = phi_1 - phi_2;
                        }
                    }
                }
                }
            }

            // Calculate average phi (from the phi's of the 5 parallel lines).
            phis[i][5] = std::vector<int>(phis[i][0].size());
            int phis_i_5_size = phis[i][5].size();
            for (int k = 0; k < phis_i_5_size; k++) {
                phis_i_5_k = phis[i][0][k] + phis[i][1][k] + phis[i][2][k] + phis[i][3][k] + phis[i][4][k];
                phis[i][5][k] = phis_i_5_k / 5;
            }

            // 获得barcode的开始结束位置 index
            // Get barcode start position.
            start_barcode_pos[i] = std::distance(phis[i][5].begin(),
                                                 std::min_element(phis[i][5].begin(),
                                                                  phis[i][5].end()));
            // Get barcode end position.
            end_barcode_pos[i] = std::distance(phis[i][5].begin(),
                                               std::max_element(phis[i][5].begin(),
                                                                phis[i][5].end()));

            cv::Mat image_debug_temp1;
            image_color.copyTo(image_debug_temp1);

            if (DEBUGCOMPHIS) {
//                for(i=0; i<start_barcode_pos.size(); i++){
////                    cv::line(image_debug_temp1, cv::Point())
//                    std::cout<<"start_barcode_pos[i]"<<start_barcode_pos[i]<<std::endl;
//                    std::cout<<"end_barcode_pos[i]"<<end_barcode_pos[i]<<std::endl;
//
//                }
                cv::Point a;
                a.x = start_barcode_pos[i];
                a.y = image_color.rows/2;
                cv::Point b;
                b.x = end_barcode_pos[i];
                b.y = image_color.rows/2;
                cv::line(image_debug_temp1, a, b, cv::Scalar(0, 255, 0), 5);
////                std::cout<<"pos kl :("<<kl->pt.x<<","<<kl->pt.y<<")"<<std::endl;
                cv::namedWindow("image_temp DEBUGCOMPHIS", cv::WINDOW_NORMAL);
                cv::imshow("image_temp DEBUGCOMPHIS", image_debug_temp1);
                cv::waitKey();
            }
        }
}

void calculateBoundingBoxes(int keylinesInContours_size,
                            std::vector<int> &start_barcode_pos,
                            std::vector<int> &end_barcode_pos,
                            std::vector<cv::line_descriptor::KeyLine> &keylines,
                            std::vector<std::vector<cv::Point>> &contours_barcode,
                            std::vector<std::vector<cv::Point>> &perpendicularLineStartEndPoints,
                            cv::Mat &image_candidates,
                            std::vector<bool> &deletedContours,
                            int index,
                            int maxLengthToLineLengthRatio,
                            int minLengthToLineLengthRatio) {

    /*
    int length;
    float angle;
    float sin_angle;
    float cos_angle;
    float keylines_i_lineLength;
    int perpendicularLineStartEndPoints_i_0_x;
    int perpendicularLineStartEndPoints_i_0_y;
    int start_barcode_pos_i;
    int end_barcode_pos_i;
    int tmp_0;
    int tmp_1;
    int tmp_2;
    int tmp_3;
    float tmp_4;
    float tmp_5;
    */

    //#pragma omp parallel for
    // Process all contours
    for (int i = 0; i < keylinesInContours_size; i++) {
        // Calculate barcode length
        int length = end_barcode_pos[i] - start_barcode_pos[i];
        // Get keyline-length (= barcode height).
        float keylines_i_lineLength = keylines[i].lineLength;//每个线段的长度
        if (false == deletedContours[i]) {
            //if(i == index) {
            // Only process candidates with a positive length.
            if (0 < length) {
                // Only process candidates with ritht ratios.
//                std::cout<<"ratio: "<<length/keylines_i_lineLength<<std::endl;
                // barcode长度/每个线段的长度
                if ((length / keylines_i_lineLength) < maxLengthToLineLengthRatio) {
                    if ((length / keylines_i_lineLength) > minLengthToLineLengthRatio) {

                        // Handle different angle cases.
                        float angle = keylines[i].angle;
                        if (0 < angle) {
                            angle = (M_PI_2 - angle);
                        } else {
                            angle = -(M_PI_2 - std::abs(angle));
                        }
                        float sin_angle = std::sin(angle);
                        float cos_angle = std::cos(angle);

                        /*
                        std::cout << "support_candidates[" << i << "] = " << support_candidates[i] << std::endl;
                        std::cout << "diff_1 = " << diff_1 << ", diff_2 = " << diff_2 << ", diff_3 = " << diff_3 << ", diff_4 = " << diff_4 << std::endl;
                        std::cout << "Add one bounding box contour!" << std::endl;
                        std::cout << "start_barcode_pos = " << start_barcode_pos[i] << " , end_barcode_pos = " << end_barcode_pos[i] << std::endl;// ", end_pos = " << phis[i][2].size() << ", angle = " << 180*angle/M_PI << std::endl;
                        std::cout << "keylines[" << i << "].lineLength = " << keylines[i].lineLength << std::endl;
                        */
                        // Calculate coordinates of the points for the barcode_contours.
                        // ???
                        int perpendicularLineStartEndPoints_i_0_x = perpendicularLineStartEndPoints[i][0].x;
                        int perpendicularLineStartEndPoints_i_0_y = perpendicularLineStartEndPoints[i][0].y;
                        int start_barcode_pos_i = start_barcode_pos[i];
                        int end_barcode_pos_i = end_barcode_pos[i];

                        int tmp_0 = perpendicularLineStartEndPoints_i_0_x + cos_angle * start_barcode_pos_i;
                        int tmp_1 = perpendicularLineStartEndPoints_i_0_y - sin_angle * start_barcode_pos_i;
                        int tmp_2 = perpendicularLineStartEndPoints_i_0_x + cos_angle * end_barcode_pos_i;
                        int tmp_3 = perpendicularLineStartEndPoints_i_0_y - sin_angle * end_barcode_pos_i;
                        float tmp_4 = keylines_i_lineLength * sin_angle * 0.5;
                        float tmp_5 = keylines_i_lineLength * cos_angle * 0.5;

                        contours_barcode[i][0] = cv::Point(tmp_0 - tmp_4,
                                                           tmp_1 - tmp_5);
                        contours_barcode[i][1] = cv::Point(tmp_2 - tmp_4,
                                                           tmp_3 - tmp_5);
                        contours_barcode[i][2] = cv::Point(tmp_2 + tmp_4,
                                                           tmp_3 + tmp_5);
                        contours_barcode[i][3] = cv::Point(tmp_0 + tmp_4,
                                                           tmp_1 + tmp_5);

                        cv::line(image_candidates, keylines[i].getStartPoint(), keylines[i].getEndPoint(),
                                 cv::Scalar(255, 0, 0), 1);
                        /*
                        cv::putText(image_candidates, std::to_string(i), contours[i][0], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));
                        std::cout << "perpencidularLineStartEndPoints[" << i << "][0].x = " << perpencidularLineStartEndPoints[i][0].x << ", perpencidularLineStartEndPoints[" << i << "][0].y = " << perpencidularLineStartEndPoints[i][0].y << std::endl;
                        std::cout << "contour[" << i << "][0] = " << contour[i][0] << ", contour[" << i << "][1] = " << contour[i][1] <<
                                                 ", contour[" << i << "][2] = " << contour[i][2] << ", contour[" << i << "][3] = " << contour[i][3] << std::endl;
                        */
                    } else { // Deactivate candidate.
                        deletedContours[i] = true;
                    }
                } else { // Deactivate candidate.
                    deletedContours[i] = true;
                }
            } else { // Deactivate candidate.
                deletedContours[i] = true;
            }
            //} // End index else
        }
    }
}

void filterContours(int keylinesInContours_size,
                    std::vector<bool> &deletedContours,
                    std::vector<int> &start_barcode_pos,
                    std::vector<int> &end_barcode_pos,
                    std::vector<cv::line_descriptor::KeyLine> &keylines,
                    std::vector<std::vector<int>> &support_scores,
                    std::vector<std::vector<cv::Point>> &contours_barcodes,
                    int inSegmentXDistance,
                    int inSegmentYDistance) {

    int length;
    int keylines_i_lineLength;
    cv::Point2f pt_i;
    cv::Point2f pt_j;

    // Process all barcode-contours
    for (int i = 0; i < keylinesInContours_size; i++) {
        // Only process if not deactivated.
        if (true == deletedContours[i]) {
            continue;
        }

        // Calculate length of barcode.
        length = end_barcode_pos[i] - start_barcode_pos[i];
        // Get keyline length (= height of barcode).
        keylines_i_lineLength = keylines[i].lineLength;
        if (false == deletedContours[i]) {
            // Go through ever keyline.
            for (int j = 0; j < keylinesInContours_size; j++) {
                // Skip the same keyline.
                if (i == j) {
                    continue;
                }
                // Skip already deactivated keylines.
                if (true == deletedContours[j]) {
                    continue;
                }

                pt_i = keylines[i].pt;
                pt_j = keylines[j].pt;

                // my adjust no use threshold
                if (support_scores[i] >= support_scores[j]) {
                    // Remove contour j
//                            std::cout<<"Remove contour j"<<std::endl;
//                            contours_barcodes[j].clear();
//                            contours_barcodes.erase(contours_barcodes.begin() + j);
                    deletedContours[j] = true;
                } else {
                    // Remove contour i
//                            std::cout<<"Remove contour i"<<std::endl;
//                            contours_barcodes[i].clear();
//                            contours_barcodes.erase(contours_barcodes.begin() + i);
                    deletedContours[i] = true;
                }

//                // If keyline is close to the current keyline (they belong to the same barcode).
//                if (std::abs(pt_i.x - pt_j.x) < inSegmentXDistance) {
//                    if (std::abs(pt_i.y - pt_j.y) < inSegmentYDistance) {
//                        // Remove the candidate with the lower score.
//                        // bug
//                        if (support_scores[i] >= support_scores[j]) {
//                            // Remove contour j
////                            std::cout<<"Remove contour j"<<std::endl;
//                            contours_barcodes[j].clear();
////                            contours_barcodes.erase(contours_barcodes.begin() + j);
//                            deletedContours[j] = true;
//                        } else {
//                            // Remove contour i
////                            std::cout<<"Remove contour i"<<std::endl;
//                            contours_barcodes[i].clear();
////                            contours_barcodes.erase(contours_barcodes.begin() + i);
//                            deletedContours[i] = true;
//                        }
//                    }
//                }
            }
        }
    }
}

cv::Point contourCenter(const std::vector<cv::Point> &contour) {
    if (0 == contour.size()) {
        return (cv::Point(-1, -1));
    }

    cv::Point contourCenter(0, 0);
    for (const auto &point : contour) {
        contourCenter += point;
    }
    contourCenter = cv::Point(contourCenter.x / contour.size(), contourCenter.y / contour.size());

    return (contourCenter);
}

std::vector<cv::Point> scaleContour(double scalingFactor,
                                    const std::vector<cv::Point> &contour,
                                    const cv::Mat &image) {
    cv::Point center = contourCenter(contour);

    std::vector<cv::Point> scaledContoursss(contour.size());
    std::transform(contour.begin(), contour.end(), scaledContoursss.begin(),
                   [&](const cv::Point &point) {
                       return scalingFactor * (point - center) + center;
                   }
    );

    return (scaledContoursss);
}

cv::Rect clamRoiToImage(cv::Rect roi, const cv::Mat &image) {
    cv::Rect clampedRoi = roi;

    if (0 > clampedRoi.x) {
        clampedRoi.x = 0;
    }
    if (image.cols < clampedRoi.y) {
        clampedRoi.y = image.cols;
    }
    if (image.cols < clampedRoi.x + clampedRoi.width) {
        clampedRoi.width = image.cols - clampedRoi.x;
    }

    if (0 > clampedRoi.y) {
        clampedRoi.y = 0;
    }
    if (image.rows < clampedRoi.y) {
        clampedRoi.y = image.rows;
    }
    if (image.rows < clampedRoi.y + clampedRoi.height) {
        clampedRoi.height = image.rows - clampedRoi.y;
    }

    return (clampedRoi);
}

//
std::vector<std::string> decodeBarcode(int keylinesInContours_size,
                                       std::vector<bool> &deletedContours,
                                       std::vector<std::vector<cv::Point>> &contours_barcodes,
                                       cv::Mat &image_greyscale,
                                       cv::Mat &image_barcodes) {

//    // Create zbar-scanner
    zbar::ImageScanner scanner;
    std::vector<cv::Point> scaledContour;
    cv::Rect roi;
    cv::Mat croppedImage;
    std::vector<std::string> barcodes;

    // Set config for zbar
    scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);

    std::vector<std::vector<cv::Point>> scaledCroppedContours;
    // Process every contour
    std::cout << "keylinesInContours_size: " << keylinesInContours_size << std::endl;
    for (int i = 0; i < keylinesInContours_size; i++) {
        // Skip deactivated candidates.
        if (true == deletedContours[i]) {
            continue;
        }

        // Scale contour and bound it to the image.
        scaledContour = scaleContour(1.5, contours_barcodes[i], image_barcodes);
        roi = cv::boundingRect(scaledContour);
        roi = clamRoiToImage(roi, image_barcodes);
        std::vector<cv::Point> scaledCroppedContour = {cv::Point(roi.x, roi.y),
                                                       cv::Point(roi.x + roi.width, roi.y),
                                                       cv::Point(roi.x + roi.width, roi.y + roi.height),
                                                       cv::Point(roi.x, roi.y + roi.height)};
        scaledCroppedContours.push_back(scaledCroppedContour);

        image_greyscale(roi).copyTo(croppedImage);
        // Set zbar image
        zbar::Image zbar_image(croppedImage.cols, croppedImage.rows, "Y800", croppedImage.data,
                               croppedImage.cols * croppedImage.rows);
        // Scan image for barcodes
        scanner.scan(zbar_image);

        // Use first detected barcode reading from image
        zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin();
        std::string barcode = symbol->get_data();
        barcodes.push_back(barcode);

        cv::putText(image_barcodes, barcode, contours_barcodes[i][0], cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(255, 0, 255));
    }
    // Draw barcode contour
    std::cout << "contours_barcodes(green): " << contours_barcodes.size() << std::endl;
//    cv::drawContours(image_barcodes, contours_barcodes, -1, cv::Scalar(0, 255, 0));
//    cv::imshow("contours_barcodes", image_barcodes);
    // Draw scaled contour (in which zbar tried to decode a barcode).
    std::cout << "scaledCroppedContours(yellow): " << contours_barcodes.size() << std::endl;
    cv::drawContours(image_barcodes, scaledCroppedContours, -1, cv::Scalar(0, 255, 255), 1);
    if (! DEBUGME){
        cv::imwrite("debug-barcodes.jpg", image_barcodes);
    }
//    cv::namedWindow("scaledCroppedContours", cv::WINDOW_NORMAL);
//    cv::imshow("scaledCroppedContours", image_barcodes);

    return (barcodes);
}
