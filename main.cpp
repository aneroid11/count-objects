#include "objectextraction.h"
#include "classification.h"
#include "utils.h"

const std::string INPUT_FILE = "../../testimages/test4.jpg";

void computeGeomParams(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours)
{
    for (const auto& cont : contours)
    {
        const double area = cv::contourArea(cont);
        const double perim = cv::arcLength(cont, true);
        const double compact = perim * perim / area;

        const cv::RotatedRect rect = cv::minAreaRect(cont);

        const double aspectRatio = rect.size.width / rect.size.height;

        std::cout << "object:\n";
        std::cout << "area: " << area << "\n";
        std::cout << "perimeter: " << perim << "\n";
        std::cout << "compactness: " << compact << "\n";
        std::cout << "aspect ratio: " << aspectRatio << "\n";
    }
}

int main()
{
    cv::Mat img = cv::imread(INPUT_FILE);
    cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);

    std::vector<std::vector<cv::Point>> contours;
    findContoursCanny(img, contours);

    std::vector<cv::Mat> objects;
    extractObjects(img, contours, objects);
//    showObjects(objects);

    computeGeomParams(contours);
    exit(0);

    rotateObjects(objects, contours);

//    compareObjects(objects[0], objects[4]);
    std::vector<std::vector<int>> objClasses;
    classifyObjects(objects, objClasses);

    drawClassification(img, contours, objClasses);
    showImg(img);

    return 0;
}
