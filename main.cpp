#include <unordered_map>

#include "objectextraction.h"
#include "classification.h"
#include "utils.h"

const std::string INPUT_FILE = "../../testimages/test3m.jpg";

void computeParams(cv::Mat& img,
                   const std::vector<std::vector<cv::Point>>& contours,
                   const std::vector<cv::Mat>& objects)
{
    for (int i = 0; i < contours.size(); i++)
    {
        const auto& cont = contours[i];
        const auto& obj = objects[i];

        cv::imshow("", obj);
        cv::waitKey();

        const double area = cv::contourArea(cont);
        const double perim = cv::arcLength(cont, true);
        const double compact = perim * perim / area;

        const cv::RotatedRect rect = cv::minAreaRect(cont);

        cv::Point2f vertices[4];
        rect.points(vertices);
        for (int j = 0; j < 4; j++)
        {
            cv::line(img, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 255, 255), 2);
        }

        const double rw = rect.size.width;
        const double rh = rect.size.height;
        const double aspectRatio = std::min(rw, rh) / std::max(rw, rh);
        const double extent = area / (rw * rh);

        std::vector<cv::Point> convexHull;
        cv::convexHull(cont, convexHull);
        const double hullArea = cv::contourArea(convexHull);
        const double solidity = area / hullArea;

        std::cout << "object:\n";
        std::cout << "area: " << area << "\n";
        std::cout << "perimeter: " << perim << "\n";
        std::cout << "compactness: " << compact << "\n";
        std::cout << "aspect ratio: " << aspectRatio << "\n";
        std::cout << "extent: " << extent << "\n";
        std::cout << "solidity: " << solidity << "\n";

//        cv::Mat data = obj.reshape(1, obj.total());
//
//        cv::calcHist()
//
//        cv::kmeans()
    }

    cv::imshow("", img);
    cv::waitKey();
}

void getSortedFrequencies(const std::vector<int>& vec, std::vector<std::pair<int, int>>& freqVec) {
    std::unordered_map<int, int> freqs;
    const int n = vec.size();
    for (int i = 0; i < n; i++) {
        freqs[vec[i]]++;
    }

    std::copy(freqs.begin(), freqs.end(), std::back_inserter<std::vector<std::pair<int, int>>>(freqVec));

    std::sort(freqVec.begin(), freqVec.end(),
              [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) -> bool
              {
                  return p1.second > p2.second;
              });
}

void computeDominantColor(const cv::Mat& img)
{
    cv::Mat data;
    img.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    const int k = 5;
    std::vector<int> labels;
    cv::Mat4f centers;
    cv::kmeans(data, k, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, centers);

    std::cout << "Dominant colors: \n";

    std::cout << centers << "\n";

//    std::cout << centers.row(getMostFrequentNumber(labels)) << "\n";
}

int main()
{
    cv::Mat img = cv::imread(INPUT_FILE);
    cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);

//    computeDominantColor(img);
//    exit(0);

    std::vector<std::vector<cv::Point>> contours;
    findContoursCanny(img, contours);

    std::vector<cv::Mat> objects;
    extractObjects(img, contours, objects);
//    showObjects(objects);

    showImg(objects[0]);
    computeDominantColor(objects[0]);
//    computeParams(img, contours, objects);
    exit(0);

    rotateObjects(objects, contours);

//    compareObjects(objects[0], objects[4]);
    std::vector<std::vector<int>> objClasses;
    classifyObjects(objects, objClasses);

    drawClassification(img, contours, objClasses);
    showImg(img);

    return 0;
}
