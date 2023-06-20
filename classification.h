#ifndef COUNT_OBJECTS_CLASSIFICATION_H
#define COUNT_OBJECTS_CLASSIFICATION_H

#include "utils.h"

class Classifier
{
public:
    Classifier(const std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours);
    Classifier(const Classifier& other) = delete;
    virtual ~Classifier() = default;

    void classify(std::vector<std::vector<int>>& classes);
    void drawClassification(cv::Mat& img, const std::vector<std::vector<int>>& objClasses);

protected:
    virtual bool _compareObjects(int o1, int o2) = 0;

    const std::vector<cv::Mat>& _objects;
    const std::vector<std::vector<cv::Point>>& _contours;
};

#endif //COUNT_OBJECTS_CLASSIFICATION_H
