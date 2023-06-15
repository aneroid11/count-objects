#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void showImg(const cv::Mat& img)
{
    cv::imshow("", img);
    cv::waitKey();
}

void findContoursOnImg(const cv::Mat& img, std::vector<std::vector<cv::Point>>& contours)
{
    cv::Mat edged;
    cv::Canny(img, edged, 100, 255);
//    showImg(edged);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {7, 7});
    cv::Mat closed;
    cv::morphologyEx(edged, closed, cv::MORPH_CLOSE, kernel);
//    showImg(kernel);
//    showImg(closed);

    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//    cv::drawContours(img, contours, -1, cv::Scalar(255, 0, 0), 4);
//    showImg(img);
}

void extractObjects(const cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours,
                    std::vector<cv::Mat>& objects)
{
    for (const auto& cont : contours)
    {
        cv::Rect bounds = cv::boundingRect(cont);
        cv::Mat newObj = cv::Mat::zeros(bounds.height, bounds.width, img.type());

        if (bounds.width < 30 && bounds.height < 30) { continue; }

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
                    newObj.at<cv::Vec4b>(y, x) = cv::Vec4b(0, 0, 0, 0);
                }
            }
        }

//        // we need a larger image to find more key points
//        cv::copyMakeBorder(newObj, newObj,
//                           100, 100, 100, 100,
//                           cv::BORDER_CONSTANT,
//                           cv::Scalar(255, 255, 255));

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

double getOrientationAngle(const std::vector<cv::Point>& contour)
{
    cv::RotatedRect rect = cv::fitEllipse(contour);
    return rect.angle;
}

void rotateImg(const cv::Mat& srcImg, cv::Mat& dstImg, const double angle)
{
    const int width = srcImg.cols;
    const int height = srcImg.rows;
    const cv::Point center = cv::Point(width / 2, height / 2);

    std::cout << angle << "\n";
    cv::Mat rotM = cv::getRotationMatrix2D(center, angle, 1.0);
//    cv::warpAffine(srcImg, dstImg, rotM, cv::Size(width * 2, height * 2));

    const cv::Rect bbox = cv::RotatedRect(cv::Point(),
                                          srcImg.size(),
                                          static_cast<float>(-angle)).boundingRect();
    // shift the center of rotation
    rotM.at<double>(0,2) += bbox.width/2.0 - srcImg.cols/2.0;
    rotM.at<double>(1,2) += bbox.height/2.0 - srcImg.rows/2.0;

    cv::warpAffine(srcImg, dstImg, rotM, cv::Size(bbox.width, bbox.height));
}

void rotateObjects(std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours)
{
    for (int i = 0; i < objects.size(); i++)
    {
        const double angle = getOrientationAngle(contours[i]);
        rotateImg(objects[i], objects[i], angle);
    }
}

int main()
{
    cv::Mat img = cv::imread("../test3.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);

    std::vector<std::vector<cv::Point>> contours;
    findContoursOnImg(img, contours);

    std::vector<cv::Mat> objects;
    extractObjects(img, contours, objects);
    rotateObjects(objects, contours);

    showObjects(objects);

    return 0;
}
