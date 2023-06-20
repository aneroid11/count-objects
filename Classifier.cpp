#include "Classifier.h"

Classifier::Classifier(const std::vector<cv::Mat> &objects, const std::vector<std::vector<cv::Point>> &contours)
    : _objects(objects), _contours(contours)
{
}

void Classifier::classify(std::vector<std::vector<int>> &classes)
{
    classes.clear();

    std::vector<int> objInds;
    for (int i = 0; i < _objects.size(); i++) { objInds.push_back(i); }

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

            if (_compareObjects(first, pairInd))
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

void Classifier::drawClassification(cv::Mat &img, const std::vector<std::vector<int>> &objClasses)
{
    for (const auto& currClass : objClasses)
    {
        const cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);

        for (const int objInd : currClass)
        {
            cv::drawContours(img, _contours, objInd, color, 3);
        }
    }
}