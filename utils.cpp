#include "utils.h"

void showImg(const cv::Mat& img)
{
    cv::imshow("", img);
    cv::waitKey();
}

double getOrientationAngle(const std::vector<cv::Point>& contour)
{
    cv::RotatedRect rect = cv::fitEllipse(contour);
    return rect.angle;
}

void rotateImg(const cv::Mat& srcImg, cv::Mat& dstImg, const double angle)
{
    const int width = srcImg.cols;
    const int height = srcImg.rows;
    const cv::Point center = cv::Point(width / 2, height / 2);

    std::cout << angle << "\n";
    cv::Mat rotM = cv::getRotationMatrix2D(center, angle, 1.0);

    const cv::Rect bbox = cv::RotatedRect(cv::Point(),
                                          srcImg.size(),
                                          static_cast<float>(-angle)).boundingRect();
    // shift the center of rotation
    rotM.at<double>(0,2) += bbox.width/2.0 - srcImg.cols/2.0;
    rotM.at<double>(1,2) += bbox.height/2.0 - srcImg.rows/2.0;

    cv::warpAffine(srcImg, dstImg, rotM, cv::Size(bbox.width, bbox.height));
//    cv::warpAffine(srcImg, dstImg, rotM, cv::Size(width, height));

    showImg(dstImg);
}