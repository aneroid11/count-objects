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

    int dstCenterX = dstImg.cols / 2;
    int dstCenterY = dstImg.rows / 2;
    double halfObjW = objW / 2;
    double halfObjH = objH / 2;
//    showImg(dstImg(cv::Rect(dstCenterX - halfObjW, dstCenterY - halfObjH, objW, objH)));
    dstImg = dstImg(cv::Rect(dstCenterX - halfObjW, dstCenterY - halfObjH, objW, objH));
}