#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::Mat blackImage(cv::Mat::zeros(512, 512, 0));
    cv::imshow("black", blackImage);
    cv::waitKey();
    return 0;
}
