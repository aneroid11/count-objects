#include "objectextraction.h"
#include "classification.h"
#include "utils.h"

const std::string INPUT_FILE = "../../testimages/test4.jpg";

void computeGeomParams(const std::vector<std::vector<cv::Point>>& contours)
{
    for (const auto& cont : contours)
    {
        const double area = cv::contourArea(cont);
        const double perim = cv::arcLength(cont, true);
        const double compact = perim * perim / area;

        cv::Moments m = cv::moments(cont);
        const double sqrt = cv::sqrt((m.m20 - m.m02) * (m.m20 - m.m02) + 4 * m.m11 * m.m11);
        const double elong = (m.m20 + m.m02 + sqrt) / (m.m20 + m.m02 - sqrt);

        std::cout << "object:\n";
        std::cout << "area: " << area << "\n";
        std::cout << "perimeter: " << perim << "\n";
        std::cout << "compactness: " << compact << "\n";
        std::cout << "elongation: " << elong << "\n";
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

    showObjects(objects);

//    computeGeomParams(contours);

    rotateObjects(objects, contours);

//    compareObjects(objects[0], objects[4]);
    std::vector<std::vector<int>> objClasses;
    classifyObjects(objects, objClasses);

    drawClassification(img, contours, objClasses);
    showImg(img);

    return 0;
}
