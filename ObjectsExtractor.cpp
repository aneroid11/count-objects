#include "ObjectsExtractor.h"
#include "utils.h"

ObjectsExtractor::ObjectsExtractor(const cv::Mat &img) : _img(img)
{
    _findContoursCanny();
    _extractObjects();
}

void ObjectsExtractor::showObjects() const
{
    for (const auto& obj : _objects)
    {
        showImg(obj);
    }
}

void ObjectsExtractor::_findContoursCanny()
{
    _contours.clear();

    cv::Mat contrasted;
    cv::convertScaleAbs(_img, contrasted, 1.3, 0);

    cv::Mat edged;
    cv::Canny(contrasted, edged, 85, 255);

    const int kernelW = _img.cols / 120;
    const int kernelH = _img.rows / 120;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {kernelW, kernelH});
    cv::Mat morphed;
//    cv::morphologyEx(edged, closed, cv::MORPH_CLOSE, kernel);
    cv::dilate(edged, morphed, kernel);

    std::vector<std::vector<cv::Point>> allContours;
    cv::findContours(morphed, allContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& cont : allContours)
    {
        cv::Rect bounds = cv::boundingRect(cont);

        if (bounds.width < 60 && bounds.height < 60)
        {
            continue;
        }
        _contours.push_back(cont);
    }
}

void ObjectsExtractor::_extractObjects()
{
    _objects.clear();

    for (const auto& cont : _contours)
    {
        cv::Rect bounds = cv::boundingRect(cont);
        cv::Mat newObj = cv::Mat::zeros(bounds.height, bounds.width, _img.type());

        for (int x = 0; x < bounds.width; x++)
        {
            for (int y = 0; y < bounds.height; y++)
            {
                // check if this point is inside the contour
                double dist = cv::pointPolygonTest(cont,
                                                   {static_cast<float>(bounds.x + x),
                                                    static_cast<float>(bounds.y + y)},
                                                   false);
                if (dist >= 0)
                {
                    newObj.at<cv::Vec3b>(y, x) = _img.at<cv::Vec3b>(y + bounds.y, x + bounds.x);
                }
                else
                {
                    newObj.at<cv::Vec3b>(y, x) = BG_COLOR;
                }
            }
        }

        _objects.push_back(newObj);
    }
}