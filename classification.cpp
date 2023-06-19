#include "classification.h"

void rotateObjects(std::vector<cv::Mat>& objects, const std::vector<std::vector<cv::Point>>& contours)
{
    for (int i = 0; i < objects.size(); i++)
    {
        cv::Size2f rectSize;
        const double angle = getOrientationAngle(contours[i], &rectSize);
        const double w = std::min(rectSize.width, rectSize.height);
        const double h = std::max(rectSize.width, rectSize.height);
        rotateImg(objects[i], objects[i], angle, w, h);
    }
}

void correctSizesForComparing(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& obj1, cv::Mat& obj2)
{
    obj1 = img1;
    obj2 = img2;

    const int w1 = obj1.cols;
    const int w2 = obj2.cols;
    const int h1 = obj1.rows;
    const int h2 = obj2.rows;

    if (w1 < w2 && h1 > h2 || w1 > w2 && h1 < h2)
    {
        const int widthToAdd = abs(w1 - w2);

        if (w1 < w2)
        {
            cv::copyMakeBorder(obj1, obj1,
                               0, 0, widthToAdd / 2, widthToAdd / 2,
                               cv::BORDER_CONSTANT,
                               BG_COLOR);
        }
        else
        {
            cv::copyMakeBorder(obj2, obj2,
                               0, 0, widthToAdd / 2, widthToAdd / 2,
                               cv::BORDER_CONSTANT,
                               BG_COLOR);
        }
    }
}

void getObjVariants(const cv::Mat& obj, std::vector<cv::Mat>& variants)
{
    variants.clear();
    variants.push_back(obj);

    cv::Mat hflipped;
    cv::flip(obj, hflipped, 1);

    cv::Mat vflipped;
    cv::flip(obj, vflipped, 0);

    cv::Mat hvflipped;
    cv::flip(obj, hvflipped, -1);

    variants.push_back(hvflipped);
    variants.push_back(vflipped);
    variants.push_back(hflipped);
}

bool compareObjects(const cv::Mat& o1, const cv::Mat& o2)
{
    const double OBJECTS_ARE_SAME_THRESHOLD = 0.65;

    cv::Mat obj1 = o1;
    cv::Mat obj2;
    cv::copyMakeBorder(o2, obj2, o1.rows/2, o1.rows/2, o1.cols/2, o1.cols/2,
                       cv::BORDER_CONSTANT, BG_COLOR);
//    correctSizesForComparing(o1, o2, obj1, obj2);

    std::vector<cv::Mat> obj1Variants;
    getObjVariants(obj1, obj1Variants);

//    cv::imwrite("o1.jpg", obj1);
//    cv::imwrite("o2.jpg", obj2);

    showImg(obj1);
    showImg(obj2);

    for (const cv::Mat& v : obj1Variants)
    {
        cv::Mat result;
//        std::cout << "before\n";
        cv::matchTemplate(v, obj2, result, cv::TM_CCOEFF_NORMED);
//        std::cout << "after\n";
//        cv::matchTemplate(v, obj2, result, cv::TM_CCORR_NORMED);
//        showImg(result);

        double maxVal;
        cv::minMaxLoc(result, nullptr, &maxVal);

//        std::cout << maxVal << "\n";

        if (maxVal > OBJECTS_ARE_SAME_THRESHOLD)
        {
            return true;
        }
    }

    return false;
}

void classifyObjects(const std::vector<cv::Mat>& objects, std::vector<std::vector<int>>& classes)
{
    classes.clear();

    std::vector<int> objInds;
    for (int i = 0; i < objects.size(); i++) { objInds.push_back(i); }

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

            std::cout << "compare:\n";

//            cv::imwrite("o1.jpg", objects[first]);
//            cv::imwrite("o2.jpg", objects[pairInd]);

//            showImg(objects[first]);
//            showImg(objects[pairInd]);

            if (compareObjects(objects[first], objects[pairInd]))
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

void drawClassification(cv::Mat& img,
                        const std::vector<std::vector<cv::Point>>& contours,
                        const std::vector<std::vector<int>>& objClasses)
{
    for (const auto& currClass : objClasses)
    {
        const cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);

        for (const int objInd : currClass)
        {
            cv::drawContours(img, contours, objInd, color, 3);
        }
    }
}