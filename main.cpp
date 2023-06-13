#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

struct Pixel
{
    char r, g, b;
};

int main()
{
    cv::Mat img = cv::imread("../test3.jpg");
//    cv::Mat img = cv::imread("../test4.png");
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

    cv::Ptr<cv::ORB> orbDetector = cv::ORB::create();

//    std::vector<cv::KeyPoint> keypoints;
//    orbDetector->detect(img, keypoints);
//    cv::drawKeypoints(img, keypoints, img, cv::Scalar(255, 0, 0));
//    cv::imshow("img", img);
//    cv::waitKey();
//    exit(1);

    std::vector<cv::Mat> objects;
    for (const auto& cont : contours)
    {
        cv::Rect bounds = cv::boundingRect(cont);
        cv::Mat newObj = cv::Mat::zeros(bounds.height, bounds.width, img.type());

        if (bounds.width < 30 && bounds.height < 30) { continue; }

        for (int x = 0; x < bounds.width; x++)
        {
            for (int y = 0; y < bounds.height; y++)
            {
//                newObj.at<Pixel>(y, x) = img.at<Pixel>(y + bounds.y, x + bounds.x);
                // check if this point is inside the contour
                double dist = cv::pointPolygonTest(cont,
                                                     {static_cast<float>(bounds.x + x),
                                                      static_cast<float>(bounds.y + y)},
                                                      false);
                if (dist >= 0)
                {
                    newObj.at<Pixel>(y, x) = img.at<Pixel>(y + bounds.y, x + bounds.x);
                }
                else
                {
                    newObj.at<Pixel>(y, x) = Pixel { static_cast<char>(255),
                                                     static_cast<char>(255),
                                                     static_cast<char>(255) };
                }
            }
        }

        cv::copyMakeBorder(newObj, newObj,
                           100, 100, 100, 100,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(255, 255, 255));

//        std::vector<cv::KeyPoint> keypoints;
//        orbDetector->detect(newObj, keypoints);
//        std::cout << "Number of keypoints: " << keypoints.size() << "\n";
//        cv::drawKeypoints(newObj, keypoints, newObj, cv::Scalar(255, 0, 0));

        cv::imshow("newObj", newObj);
        cv::waitKey();

        objects.push_back(newObj);
    }

    std::cout << "total objects: " << objects.size() << "\n";

    return 0;
}
