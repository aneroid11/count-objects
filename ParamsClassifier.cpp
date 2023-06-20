#include "ParamsClassifier.h"

ParamsClassifier::ParamsClassifier(const std::vector<cv::Mat>& objects,
                                   const std::vector<std::vector<cv::Point>>& contours)
                                   : Classifier(objects, contours)
{
    _computeParams();
}

void ParamsClassifier::_computeParams()
{
    _params.resize(_contours.size());

    for (int i = 0; i < _contours.size(); i++)
    {
        const auto& cont = _contours[i];
        const auto& obj = _objects[i];

        const double area = cv::contourArea(cont);
        const double perim = cv::arcLength(cont, true);
        const double compact = perim * perim / area;

        const cv::RotatedRect rect = cv::minAreaRect(cont);

        const double rw = rect.size.width;
        const double rh = rect.size.height;
        const double aspectRatio = std::min(rw, rh) / std::max(rw, rh);
        const double extent = area / (rw * rh);

        std::vector<cv::Point> convexHull;
        cv::convexHull(cont, convexHull);
        const double hullArea = cv::contourArea(convexHull);
        const double solidity = area / hullArea;

        _params[i].area = area;
        _params[i].perim = perim;
        _params[i].compact = compact;
        _params[i].solidity = solidity;
        _params[i].aspectRatio = aspectRatio;
        _params[i].extent = extent;

        const cv::Vec3f domColor = computeDominantColor(obj);
        _params[i].domB = domColor.val[0];
        _params[i].domG = domColor.val[1];
        _params[i].domR = domColor.val[2];
    }
}

bool ParamsClassifier::_compareObjects(int o1ind, int o2ind)
{
    const ObjectParams& o1 = _params[o1ind];
    const ObjectParams& o2 = _params[o2ind];

    if (std::min(o1.area, o2.area) / std::max(o1.area, o2.area) < 0.85) { return false; }
    if (std::min(o1.perim, o2.perim) / std::max(o1.perim, o2.perim) < 0.85) { return false; }
    if (std::min(o1.compact, o2.compact) / std::max(o1.compact, o2.compact) < 0.8) { return false; }
    if (std::abs(o1.extent - o2.extent) > 0.1) { return false; }
    if (std::abs(o1.aspectRatio - o2.aspectRatio) > 0.1) { return false; }
    if (std::abs(o1.solidity - o2.solidity) > 0.1) { return false; }
    if (std::abs(o1.domR - o2.domR) > 80) { return false; }
    if (std::abs(o1.domG - o2.domG) > 80) { return false; }
    if (std::abs(o1.domB - o2.domB) > 80) { return false; }

    return true;
}
