#ifndef COUNT_OBJECTS_PARAMSCLASSIFIER_H
#define COUNT_OBJECTS_PARAMSCLASSIFIER_H

#include "classification.h"

class ParamsClassifier : public Classifier
{
public:
    ParamsClassifier(const std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours);

protected:
    bool _compareObjects(int o1, int o2) override;
    void _computeParams();

private:
    struct ObjectParams
    {
        double area, perim, compact;
        double aspectRatio;
        double extent;
        double solidity;
        double domR, domG, domB;
    };

    std::vector<ObjectParams> _params;
};


#endif //COUNT_OBJECTS_PARAMSCLASSIFIER_H
