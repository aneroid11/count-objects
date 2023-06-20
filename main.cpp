#include <chrono>

#include "ObjectsExtractor.h"
#include "TemplateMatchClassifier.h"
#include "ParamsClassifier.h"

void testMatchTemplate()
{
    cv::Mat o1 = cv::imread("../../testimages/what/o1.jpg");
    cv::Mat o2 = cv::imread("../../testimages/what/o2.jpg");

    cv::flip(o1, o1, -1);

//    const int o1width = o1.cols;
//    const int o1height = o1.rows;
//    cv::copyMakeBorder(o2, o2, o1height/2, o1height/2, o1width/2, o1width/2,
//                       cv::BORDER_CONSTANT,
//                       BG_COLOR);
    showImg(o1);
    showImg(o2);

    cv::Mat res;
    cv::matchTemplate(o1, o2, res, cv::TM_SQDIFF_NORMED);

    double min, max;
    cv::minMaxLoc(res, &min, &max);
    std::cout << "min: " << min << "\n";
    std::cout << "max: " << max << "\n";

    showImg(res);

    exit(0);
}

int main()
{
//    testMatchTemplate();

    const std::string INPUT_FILE = "../../testimages/test4.jpg";

    srand(time(nullptr));

    cv::Mat img = cv::imread(INPUT_FILE);

    std::unique_ptr<ObjectsExtractor> extractor(new ObjectsExtractor(img));
    const std::vector<std::vector<cv::Point>>& contours = extractor->getContours();
    const std::vector<cv::Mat>& objects = extractor->getObjects();
    extractor->showObjects();

    std::vector<std::vector<int>> objClasses;

    auto begin = std::chrono::system_clock::now();

//    std::unique_ptr<Classifier> classifier(new TemplateMatchClassifier(objects, contours));
    std::unique_ptr<Classifier> classifier(new ParamsClassifier(objects, contours));
    classifier->classify(objClasses);

    auto end = std::chrono::system_clock::now();
    auto deltaTime = end - begin;
    std::cout << std::chrono::duration<double>(deltaTime).count() << " seconds\n";

    classifier->drawClassification(img, objClasses);
    showImg(img);

    cv::imwrite("output.jpg", img);

    return 0;
}
