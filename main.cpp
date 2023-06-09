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

    cv::imshow("img", img);
    cv::waitKey();
    return 0;
}
