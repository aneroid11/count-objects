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
    showImg(edged);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {7, 7});
    cv::Mat closed;
    cv::morphologyEx(edged, closed, cv::MORPH_CLOSE, kernel);
//    showImg(kernel);
    showImg(closed);

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
                    newObj.at<int32_t>(y, x) = img.at<int32_t>(y + bounds.y, x + bounds.x);
                }
                /*else
                {
                    newObj.at<cv::Scalar>(y, x) = cv::Scalar(255, 255, 255, 0);
                }*/
            }
        }
//
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

int main()
{
    cv::Mat img = cv::imread("../test3.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);

    std::vector<std::vector<cv::Point>> contours;
    findContoursOnImg(img, contours);

    std::vector<cv::Mat> objects;
    extractObjects(img, contours, objects);
    showObjects(objects);

//    cv::Mat imgGray;
//    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
//
//    cv::Mat binaryImg;
//    cv::threshold(imgGray, binaryImg, 120, 255, cv::THRESH_BINARY_INV);
//    showImg(binaryImg);
//
//    // closing
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
//    cv::Mat closedImg;
//    cv::morphologyEx(binaryImg, closedImg, cv::MORPH_CLOSE, kernel);
//    showImg(closedImg);
//
//    exit(0);
/*
    cv::Mat edged;
    cv::Canny(img, edged, 10, 250);

    cv::imshow("window", edged);
    cv::waitKey();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {7, 7});
    cv::Mat closed;
    cv::morphologyEx(edged, closed, cv::MORPH_CLOSE, kernel);

//    cv::imshow("window", closed);
//    cv::waitKey();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::drawContours(img, contours, -1, cv::Scalar(255, 0, 0), 4);
    cv::imshow("img", img);
    cv::waitKey();
//    exit(0);

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
//                    newObj.at<Pixel>(y, x) = img.at<Pixel>(y + bounds.y, x + bounds.x);
                    newObj.at<char>(y, x) = img.at<char>(y + bounds.y, x + bounds.x);
                }
                else
                {
                    newObj.at<char>(y, x) = static_cast<char>(255);
                }
            }
        }

        // we need a larger image to find more key points
        cv::copyMakeBorder(newObj, newObj,
                           100, 100, 100, 100,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(255, 255, 255));

//        std::vector<cv::KeyPoint> keypoints;
//        orbDetector->detect(newObj, keypoints);
//        cv::drawKeypoints(newObj, keypoints, newObj, cv::Scalar(255, 0, 0));
//
//        cv::imshow("newObj", newObj);
//        cv::waitKey();

        objects.push_back(newObj);
    }

    std::cout << "total objects: " << objects.size() << "\n";

    std::vector<std::vector<cv::KeyPoint>> objectsKeypoints;
    std::vector<cv::Mat> objectsDescriptors;

    // find key points and descriptors for all the objects
    for (int i = 0; i < objects.size(); i++)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orbDetector->detectAndCompute(objects[i], cv::noArray(),
                                      keypoints, descriptors);

        std::cout << "Number of keypoints: " << keypoints.size() << "\n";

        objectsKeypoints.push_back(keypoints);
        objectsDescriptors.push_back(descriptors);
    }

    std::vector<cv::KeyPoint> imgKeypoints;
    cv::Mat imgDescriptors;
    orbDetector->detectAndCompute(img, cv::noArray(), imgKeypoints, imgDescriptors);

    cv::BFMatcher bfmatcher(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> matches;
    const int O1 = 2;
    const int O2 = 1;
    bfmatcher.match(objectsDescriptors[O1], objectsDescriptors[O2], matches);
//    bfmatcher.match(objectsDescriptors[O1], imgDescriptors, matches);

    std::cout << matches.size() << "\n";

    std::sort(matches.begin(),
              matches.end(),
              [](const cv::DMatch& m1, const cv::DMatch& m2){ return m1.distance < m2.distance; });

//    std::vector<cv::DMatch> matchesToDraw = matches;
    std::vector<cv::DMatch> matchesToDraw;
    for (int i = 0; i < 500 && i < matches.size(); i++)
    {
        matchesToDraw.push_back(matches[i]);
    }

    cv::Mat outImg;
//    cv::drawMatches(objects[O1], objectsKeypoints[O1], img, imgKeypoints,
//                    matchesToDraw, outImg, cv::Scalar(255, 0, 0));
    cv::drawMatches(objects[O1], objectsKeypoints[O1], objects[O2], objectsKeypoints[O2],
                    matchesToDraw, outImg, cv::Scalar(255, 0, 0));

    cv::imshow("outImg", outImg);
    cv::waitKey();
//    std::cout << "object 0 and 2: matches: " << matches.size() << "\n";
//    cv::drawMatches(objec)
*/
    return 0;
}
