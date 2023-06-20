#ifndef COUNT_OBJECTS_OBJECTSEXTRACTOR_H
#define COUNT_OBJECTS_OBJECTSEXTRACTOR_H

#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ObjectsExtractor
{
public:
    ObjectsExtractor(const cv::Mat& img);
    ObjectsExtractor(const ObjectsExtractor& other) = delete;

    const std::vector<std::vector<cv::Point>>& getContours() const { return _contours; }
    const std::vector<cv::Mat>& getObjects() const { return _objects; }

    void showObjects() const;

private:
    void _findContoursCanny();
    void _extractObjects();

    const cv::Mat& _img;
    std::vector<cv::Mat> _objects;
    std::vector<std::vector<cv::Point>> _contours;
};

#endif //COUNT_OBJECTS_OBJECTSEXTRACTOR_H
