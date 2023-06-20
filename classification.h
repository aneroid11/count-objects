#ifndef COUNT_OBJECTS_CLASSIFICATION_H
#define COUNT_OBJECTS_CLASSIFICATION_H

#include "utils.h"

class Classifier
{
public:
    Classifier(const std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours);
    Classifier(const Classifier& other) = delete;

    void classify(std::vector<std::vector<int>>& classes);
    void drawClassification(cv::Mat& img, const std::vector<std::vector<int>>& objClasses);

protected:
    virtual bool _compareObjects(int o1, int o2) = 0;

    const std::vector<cv::Mat>& _objects;
    const std::vector<std::vector<cv::Point>>& _contours;
};

void rotateObjects(std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours);
//void correctSizesForComparing(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& obj1, cv::Mat& obj2);
void getObjVariants(const cv::Mat& obj, std::vector<cv::Mat>& variants);
bool compareObjects(const cv::Mat& o1, const cv::Mat& o2);
void classifyObjects(const std::vector<cv::Mat>& objects, std::vector<std::vector<int>>& classes);
void drawClassification(cv::Mat& img,
                        const std::vector<std::vector<cv::Point>>& contours,
                        const std::vector<std::vector<int>>& objClasses);

#endif //COUNT_OBJECTS_CLASSIFICATION_H
