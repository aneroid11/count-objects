#ifndef COUNT_OBJECTS_UTILS_H
#define COUNT_OBJECTS_UTILS_H

#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void showImg(const cv::Mat& img);
double getOrientationAngle(const std::vector<cv::Point>& contour);
void rotateImg(const cv::Mat& srcImg, cv::Mat& dstImg, double angle);

#endif //COUNT_OBJECTS_UTILS_H
