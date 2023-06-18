#ifndef COUNT_OBJECTS_CLASSIFICATION_H
#define COUNT_OBJECTS_CLASSIFICATION_H

#include "utils.h"

void rotateObjects(std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours);
void correctSizesForComparing(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& obj1, cv::Mat& obj2);
void getObjVariants(const cv::Mat& obj, std::vector<cv::Mat>& variants);
bool compareObjects(const cv::Mat& o1, const cv::Mat& o2);
void classifyObjects(const std::vector<cv::Mat>& objects, std::vector<std::vector<int>>& classes);
void drawClassification(cv::Mat& img,
                        const std::vector<std::vector<cv::Point>>& contours,
                        const std::vector<std::vector<int>>& objClasses);

#endif //COUNT_OBJECTS_CLASSIFICATION_H
