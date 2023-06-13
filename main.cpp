#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct Pixel
{
    char r, g, b;
};

int main()
{
    cv::Mat img = cv::imread("../test3.jpg");
//    cv::Mat img = cv::imread("../test.png");
    cv::Mat edged;
    cv::Canny(img, edged, 10, 250);

//    cv::imshow("window", edged);
//    cv::waitKey();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {7, 7});
    cv::Mat closed;
    cv::morphologyEx(edged, closed, cv::MORPH_CLOSE, kernel);

//    cv::imshow("window", closed);
//    cv::waitKey();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

//    cv::drawContours(img, contours, -1, cv::Scalar(255, 0, 0), 4);

    std::vector<cv::Mat> objects;
    for (const auto& cont : contours)
    {
        cv::Rect bounds = cv::boundingRect(cont);
        cv::Mat newObj = cv::Mat::zeros(bounds.height, bounds.width, img.type());

        std::cout << img.type() << "\n";

        if (bounds.width < 30 && bounds.height < 30) { continue; }

        for (int x = 0; x < bounds.width; x++)
        {
            for (int y = 0; y < bounds.height; y++)
            {
                // check if this point is inside the contour
                double result = cv::pointPolygonTest(cont,
                                                     {static_cast<float>(bounds.x + x),
                                                      static_cast<float>(bounds.y + y)},
                                                      false);
                if (result >= 0) // inside or on the contour
                {
                    newObj.at<Pixel>(y, x) = img.at<Pixel>(y + bounds.y, x + bounds.x);
                }
                else
                {
                    // outside
//                    newObj.at<cv::Scalar>(y, x) = cv::Scalar(255, 255, 255);
                }
            }
        }

        cv::imshow("newObj", newObj);
        cv::waitKey();

        objects.push_back(newObj);
    }

    std::cout << "total objects: " << objects.size() << "\n";

    return 0;
}
