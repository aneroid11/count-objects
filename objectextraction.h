#ifndef COUNT_OBJECTS_OBJECTEXTRACTION_H
#define COUNT_OBJECTS_OBJECTEXTRACTION_H

#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void findContoursCanny(const cv::Mat& img, std::vector<std::vector<cv::Point>>& contours);
void extractObjects(const cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours,
                    std::vector<cv::Mat>& objects, const cv::Vec4b& bgColor);
void showObjects(const std::vector<cv::Mat>& objects);

#endif //COUNT_OBJECTS_OBJECTEXTRACTION_H
