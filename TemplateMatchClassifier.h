#ifndef COUNT_OBJECTS_TEMPLATEMATCHCLASSIFIER_H
#define COUNT_OBJECTS_TEMPLATEMATCHCLASSIFIER_H

#include "classification.h"

class TemplateMatchClassifier : public Classifier {
public:
    TemplateMatchClassifier(const std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours);

protected:
    bool _compareObjects(int o1, int o2) override;

private:
    void _rotateObjects();
    void _getObjVariants(const cv::Mat& obj, std::vector<cv::Mat>& variants);

    std::vector<cv::Mat> _rotatedObjects;
};


#endif //COUNT_OBJECTS_TEMPLATEMATCHCLASSIFIER_H
