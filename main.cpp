#include <unordered_map>
#include <chrono>

#include "objectextraction.h"
#include "utils.h"
#include "TemplateMatchClassifier.h"
#include "ParamsClassifier.h"

//void classifyObjectsByParams(const std::vector<cv::Mat>& objects, const std::vector<ObjectParams>& params, std::vector<std::vector<int>>& classes)
//{
//    // TODO: remove this duplication
//
//    classes.clear();
//
//    std::vector<int> objInds;
//    for (int i = 0; i < params.size(); i++) { objInds.push_back(i); }
//
//    while (true)
//    {
//        if (objInds.empty())
//        {
//            return;
//        }
//        if (objInds.size() < 2)
//        {
//            classes.push_back(objInds);
//            return;
//        }
//
//        std::vector<int> currClass;
//        const int first = objInds[0];
//
//        auto position = std::find(objInds.begin(), objInds.end(), first);
//        objInds.erase(position);
//
//        currClass.push_back(first);
//
//        int i = 0;
//        while (i < objInds.size())
//        {
//            const int pairInd = objInds[i];
//
////            showImg(objects[first]);
////            showImg(objects[pairInd]);
//
//            if (compareObjects(params[first], params[pairInd]))
//            {
//                position = std::find(objInds.begin(), objInds.end(), pairInd);
//                objInds.erase(position);
//
//                currClass.push_back(pairInd);
//                continue;
//            }
//            i++;
//        }
//
//        classes.push_back(currClass);
//    }
//}
//
//void classifyUsingObjParams(const std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours,
//                            std::vector<std::vector<int>>& objClasses)
//{
//    std::vector<ObjectParams> params(objects.size());
//    computeParams(contours, objects, params);
//    classifyObjectsByParams(objects, params, objClasses);
//}

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

//    const std::string INPUT_FILE = "../../testimages/testmila_m.jpg";
//    const std::string INPUT_FILE = "../../testimages/whitebg.jpg";
    const std::string INPUT_FILE = "../../testimages/test4.jpg";

    srand(time(nullptr));

    cv::Mat img = cv::imread(INPUT_FILE);

    std::vector<std::vector<cv::Point>> contours;
    findContoursCanny(img, contours);

    std::vector<cv::Mat> objects;
    extractObjects(img, contours, objects);
//    showObjects(objects);

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
