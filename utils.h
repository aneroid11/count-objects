#ifndef COUNT_OBJECTS_UTILS_H
#define COUNT_OBJECTS_UTILS_H

#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const cv::Vec3b BG_COLOR = cv::Vec3b(0, 0, 0);

void showImg(const cv::Mat& img);
double getOrientationAngle(const std::vector<cv::Point>& contour, cv::Size2f* rectSize = nullptr);
void rotateImg(const cv::Mat& srcImg, cv::Mat& dstImg, double angle, double objW, double objH);

void getSortedFrequencies(const std::vector<int>& vec, std::vector<std::pair<int, int>>& freqVec);
cv::Vec3f computeDominantColor(const cv::Mat& img);

#endif //COUNT_OBJECTS_UTILS_H
