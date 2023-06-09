#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::Mat img = cv::imread("/home/lucky/dev/6sem/practice/count-objects/test.jpg");
    if (img.empty())
    {
        std::cerr << "failed to read image\n";
        exit(EXIT_FAILURE);
    }

    cv::circle(img, {50, 50}, 30, cv::Scalar(200, 100, 200), 5);

    cv::imshow("img", img);
    cv::waitKey();
    return 0;
}
