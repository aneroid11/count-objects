#include "objectextraction.h"
#include "classification.h"
#include "utils.h"

const std::string INPUT_FILE = "../../testimages/test3m.jpg";

void computeGeomParams(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours)
{
    for (const auto& cont : contours)
    {
        const double area = cv::contourArea(cont);
        const double perim = cv::arcLength(cont, true);
        const double compact = perim * perim / area;

        const cv::RotatedRect rect = cv::minAreaRect(cont);

        cv::Point2f vertices[4];
        rect.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 255, 255), 2);
        }

        const double rw = rect.size.width;
        const double rh = rect.size.height;
        const double aspectRatio = std::min(rw, rh) / std::max(rw, rh);
        const double extent = area / (rw * rh);

        std::cout << "object:\n";
        std::cout << "area: " << area << "\n";
        std::cout << "perimeter: " << perim << "\n";
        std::cout << "compactness: " << compact << "\n";
        std::cout << "aspect ratio: " << aspectRatio << "\n";
        std::cout << "extent: " << extent << "\n";
    }

    cv::imshow("", img);
    cv::waitKey();
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

    computeGeomParams(img, contours);
    exit(0);

    rotateObjects(objects, contours);

//    compareObjects(objects[0], objects[4]);
    std::vector<std::vector<int>> objClasses;
    classifyObjects(objects, objClasses);

    drawClassification(img, contours, objClasses);
    showImg(img);

    return 0;
}
