#include "objectextraction.h"
#include "utils.h"

void findContoursCanny(const cv::Mat& img, std::vector<std::vector<cv::Point>>& contours)
{
    contours.clear();

    cv::Mat contrasted;
    cv::convertScaleAbs(img, contrasted, 1.3, 0);
    showImg(contrasted);

    cv::Mat edged;
    cv::Canny(contrasted, edged, 85, 255);
    showImg(edged);

    const int kernelW = img.cols / 120;
    const int kernelH = img.rows / 120;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {kernelW, kernelH});
    cv::Mat morphed;
//    cv::morphologyEx(edged, closed, cv::MORPH_CLOSE, kernel);
    cv::dilate(edged, morphed, kernel);
    showImg(morphed);

    std::vector<std::vector<cv::Point>> allContours;
    cv::findContours(morphed, allContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& cont : allContours)
    {
        cv::Rect bounds = cv::boundingRect(cont);

        if (bounds.width < 60 && bounds.height < 60)
        {
            continue;
        }
        contours.push_back(cont);
    }
}

void extractObjects(const cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours,
                    std::vector<cv::Mat>& objects, const cv::Vec4b& bgColor)
{
    for (const auto& cont : contours)
    {
        cv::Rect bounds = cv::boundingRect(cont);
        cv::Mat newObj = cv::Mat::zeros(bounds.height, bounds.width, img.type());

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
                    newObj.at<cv::Vec4b>(y, x) = img.at<cv::Vec4b>(y + bounds.y, x + bounds.x);
                }
                else
                {
                    newObj.at<cv::Vec4b>(y, x) = bgColor;
                }
            }
        }

        objects.push_back(newObj);
    }
}

void showObjects(const std::vector<cv::Mat>& objects)
{
    for (const auto& obj : objects)
    {
        showImg(obj);
    }
}