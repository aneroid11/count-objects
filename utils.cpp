#include "utils.h"

void showImg(const cv::Mat& img)
{
    cv::imshow("", img);
    cv::waitKey();
}

double getOrientationAngle(const std::vector<cv::Point>& contour, cv::Size2f* rectSize)
{
    cv::RotatedRect rect = cv::fitEllipse(contour);
    *rectSize = rect.size;
    return rect.angle;
}

void rotateImg(const cv::Mat& srcImg, cv::Mat& dstImg, double angle, double objW, double objH)
{
    const int width = srcImg.cols;
    const int height = srcImg.rows;
    const cv::Point center = cv::Point(width / 2, height / 2);

    cv::Mat rotM = cv::getRotationMatrix2D(center, angle, 1.0);

    const cv::Rect bbox = cv::RotatedRect(cv::Point(),
                                          srcImg.size(),
                                          static_cast<float>(-angle)).boundingRect();
    // shift the center of rotation
    rotM.at<double>(0,2) += bbox.width/2.0 - srcImg.cols/2.0;
    rotM.at<double>(1,2) += bbox.height/2.0 - srcImg.rows/2.0;

    cv::warpAffine(srcImg, dstImg, rotM, cv::Size(bbox.width, bbox.height));
//    cv::warpAffine(srcImg, dstImg, rotM, cv::Size(width, height));

    // erase the black borders

    if (objW > dstImg.cols) { objW = dstImg.cols; }
    if (objH > dstImg.rows) { objH = dstImg.rows; }

    int dstCenterX = dstImg.cols / 2;
    int dstCenterY = dstImg.rows / 2;
    double halfObjW = objW / 2;
    double halfObjH = objH / 2;

    int leftX = dstCenterX - halfObjW;
    int topY = dstCenterY - halfObjH;
    dstImg = dstImg(cv::Rect(leftX, topY, objW, objH));

//    showImg(dstImg);
}

void getSortedFrequencies(const std::vector<int>& vec, std::vector<std::pair<int, int>>& freqVec)
{
    std::unordered_map<int, int> freqs;
    const int n = vec.size();
    for (int i = 0; i < n; i++) {
        freqs[vec[i]]++;
    }

    std::copy(freqs.begin(), freqs.end(), std::back_inserter<std::vector<std::pair<int, int>>>(freqVec));

    std::sort(freqVec.begin(), freqVec.end(),
              [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) -> bool
              {
                  return p1.second > p2.second;
              });
}

cv::Vec3f computeDominantColor(const cv::Mat& img)
{
    cv::Mat data;
    img.convertTo(data, CV_32F);

    data = data.reshape(1, data.total());

    const int k = 5;
    std::vector<int> labels;
    cv::Mat3f centers;
    cv::kmeans(data, k, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, centers);

    std::vector<std::pair<int, int>> freqVec;
    getSortedFrequencies(labels, freqVec);

    cv::Vec3f mostFrequent = centers.row(freqVec[0].first).at<cv::Vec3f>();

//    const cv::Vec3f bg(BG_COLOR.val[0], BG_COLOR.val[1], BG_COLOR.val[2]);
    if (mostFrequent.val[0] < 2 && mostFrequent.val[1] < 2 && mostFrequent.val[2] < 2)
    {
        mostFrequent = centers.row(freqVec[1].first).at<cv::Vec3f>();
    }

    return mostFrequent;
}