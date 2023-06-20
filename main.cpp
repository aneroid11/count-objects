#include <unordered_map>

#include "objectextraction.h"
#include "classification.h"
#include "utils.h"

struct ObjectParams
{
    double area, perim, compact;
    double aspectRatio;
    double extent;
    double solidity;
    double domR, domG, domB;
};

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

cv::Vec3f computeDominantColor(const cv::Mat& img)
{
    cv::Mat data;
    img.convertTo(data, CV_32F);

    data = data.reshape(1, data.total());

    const int k = 5;
    std::vector<int> labels;
    cv::Mat3f centers;
    cv::kmeans(data, k, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, centers);

    std::vector<std::pair<int, int>> freqVec;
    getSortedFrequencies(labels, freqVec);

    cv::Vec3f mostFrequent = centers.row(freqVec[0].first).at<cv::Vec3f>();

//    const cv::Vec3f bg(BG_COLOR.val[0], BG_COLOR.val[1], BG_COLOR.val[2]);
    if (mostFrequent.val[0] < 2 && mostFrequent.val[1] < 2 && mostFrequent.val[2] < 2)
    {
        mostFrequent = centers.row(freqVec[1].first).at<cv::Vec3f>();
    }

    return mostFrequent;
}

void computeParams(const std::vector<std::vector<cv::Point>>& contours,
                   const std::vector<cv::Mat>& objects,
                   std::vector<ObjectParams>& params)
{
    for (int i = 0; i < contours.size(); i++)
    {
        const auto& cont = contours[i];
        const auto& obj = objects[i];

        const double area = cv::contourArea(cont);
        const double perim = cv::arcLength(cont, true);
        const double compact = perim * perim / area;

        const cv::RotatedRect rect = cv::minAreaRect(cont);

        const double rw = rect.size.width;
        const double rh = rect.size.height;
        const double aspectRatio = std::min(rw, rh) / std::max(rw, rh);
        const double extent = area / (rw * rh);

        std::vector<cv::Point> convexHull;
        cv::convexHull(cont, convexHull);
        const double hullArea = cv::contourArea(convexHull);
        const double solidity = area / hullArea;

        params[i].area = area;
        params[i].perim = perim;
        params[i].compact = compact;
        params[i].solidity = solidity;
        params[i].aspectRatio = aspectRatio;
        params[i].extent = extent;

        const cv::Vec3f domColor = computeDominantColor(obj);
        params[i].domB = domColor.val[0];
        params[i].domG = domColor.val[1];
        params[i].domR = domColor.val[2];
    }
}

void classifyUsingTemplateMatching(std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours,
                                   std::vector<std::vector<int>>& objClasses)
{
    rotateObjects(objects, contours);
    classifyObjects(objects, objClasses);
}

bool compareObjects(const ObjectParams& o1, const ObjectParams& o2)
{
    if (std::min(o1.area, o2.area) / std::max(o1.area, o2.area) < 0.85) { return false; }
    if (std::min(o1.perim, o2.perim) / std::max(o1.perim, o2.perim) < 0.85) { return false; }
    if (std::min(o1.compact, o2.compact) / std::max(o1.compact, o2.compact) < 0.8) { return false; }
    if (std::abs(o1.extent - o2.extent) > 0.1) { return false; }
    if (std::abs(o1.aspectRatio - o2.aspectRatio) > 0.1) { return false; }
    if (std::abs(o1.solidity - o2.solidity) > 0.1) { return false; }
    if (std::abs(o1.domR - o2.domR) > 80) { return false; }
    if (std::abs(o1.domG - o2.domG) > 80) { return false; }
    if (std::abs(o1.domB - o2.domB) > 80) { return false; }

    return true;
}

void classifyObjectsByParams(const std::vector<cv::Mat>& objects, const std::vector<ObjectParams>& params, std::vector<std::vector<int>>& classes)
{
    // TODO: remove this duplication

    classes.clear();

    std::vector<int> objInds;
    for (int i = 0; i < params.size(); i++) { objInds.push_back(i); }

    while (true)
    {
        if (objInds.empty())
        {
            return;
        }
        if (objInds.size() < 2)
        {
            classes.push_back(objInds);
            return;
        }

        std::vector<int> currClass;
        const int first = objInds[0];

        auto position = std::find(objInds.begin(), objInds.end(), first);
        objInds.erase(position);

        currClass.push_back(first);

        int i = 0;
        while (i < objInds.size())
        {
            const int pairInd = objInds[i];

//            showImg(objects[first]);
//            showImg(objects[pairInd]);

            if (compareObjects(params[first], params[pairInd]))
            {
                position = std::find(objInds.begin(), objInds.end(), pairInd);
                objInds.erase(position);

                currClass.push_back(pairInd);
                continue;
            }
            i++;
        }

        classes.push_back(currClass);
    }
}

void classifyUsingObjParams(const std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours,
                            std::vector<std::vector<int>>& objClasses)
{
    std::vector<ObjectParams> params(objects.size());
    computeParams(contours, objects, params);
    classifyObjectsByParams(objects, params, objClasses);
}

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
    const std::string INPUT_FILE = "../../testimages/test2.jpg";

    srand(time(nullptr));

    cv::Mat img = cv::imread(INPUT_FILE);
//    cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);

    std::vector<std::vector<cv::Point>> contours;
    findContoursCanny(img, contours);

    std::vector<cv::Mat> objects;
    extractObjects(img, contours, objects);
//    showObjects(objects);

    std::vector<std::vector<int>> objClasses;

    // std::unique_ptr<Classifier> classifier = new ParamsClassifier(objects, contours);
    // classifier.classify(objClasses);
    classifyUsingTemplateMatching(objects, contours, objClasses);
//    classifyUsingObjParams(objects, contours, objClasses);

    drawClassification(img, contours, objClasses);
    showImg(img);

    cv::imwrite("output.jpg", img);

    return 0;
}
