#include "TemplateMatchClassifier.h"

TemplateMatchClassifier::TemplateMatchClassifier(const std::vector<cv::Mat>& objects,
                                                 const std::vector<std::vector<cv::Point>>& contours)
                                                 : Classifier(objects, contours)
{
    _rotateObjects();
}

void TemplateMatchClassifier::_rotateObjects()
{
    _rotatedObjects.resize(_objects.size());

    for (int i = 0; i < _objects.size(); i++)
    {
        cv::Size2f rectSize;
        double angle = getOrientationAngle(_contours[i], &rectSize);
        double w = std::min(rectSize.width, rectSize.height);
        double h = std::max(rectSize.width, rectSize.height);
        rotateImg(_objects[i], _rotatedObjects[i], angle, w, h);
    }
}

void TemplateMatchClassifier::_getObjVariants(const cv::Mat& obj, std::vector<cv::Mat>& variants)
{
    variants.clear();
    variants.push_back(obj);

    cv::Mat hflipped;
    cv::flip(obj, hflipped, 1);

    cv::Mat vflipped;
    cv::flip(obj, vflipped, 0);

    cv::Mat hvflipped;
    cv::flip(obj, hvflipped, -1);

    variants.push_back(hvflipped);
    variants.push_back(vflipped);
    variants.push_back(hflipped);
}

bool TemplateMatchClassifier::_compareObjects(int o1index, int o2index)
{
    const double OBJECTS_ARE_SAME_THRESHOLD = 0.75;
    const cv::Mat& o1 = _rotatedObjects[o1index];
    const cv::Mat& o2 = _rotatedObjects[o2index];

    cv::Mat obj1 = o1;
    cv::Mat obj2;
    cv::copyMakeBorder(o2, obj2, o1.rows/2, o1.rows/2, o1.cols/2, o1.cols/2,
                       cv::BORDER_CONSTANT, BG_COLOR);

    std::vector<cv::Mat> obj1Variants;
    _getObjVariants(obj1, obj1Variants);

    for (const cv::Mat& v : obj1Variants)
    {
        cv::Mat result;
//        cv::matchTemplate(v, obj2, result, cv::TM_CCOEFF_NORMED);
//        cv::matchTemplate(v, obj2, result, cv::TM_SQDIFF_NORMED);
        cv::matchTemplate(v, obj2, result, cv::TM_CCORR_NORMED);

        double maxVal, minVal;
        cv::minMaxLoc(result, &minVal, &maxVal);

//        std::cout << "min val: " << minVal << "\n";
//        std::cout << "max val: " << maxVal << "\n\n";

        if (maxVal > OBJECTS_ARE_SAME_THRESHOLD)
//        if (minVal < 0.4)
        {
            return true;
        }
    }

    return false;
}