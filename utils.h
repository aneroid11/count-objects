#ifndef COUNT_OBJECTS_UTILS_H
#define COUNT_OBJECTS_UTILS_H

#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const cv::Vec4b BG_COLOR = cv::Vec4b(0, 0, 0, 0);

void showImg(const cv::Mat& img);
double getOrientationAngle(const std::vector<cv::Point>& contour, cv::Size2f* rectSize = nullptr);
void rotateImg(const cv::Mat& srcImg, cv::Mat& dstImg, double angle, double objW, double objH);

#endif //COUNT_OBJECTS_UTILS_H
