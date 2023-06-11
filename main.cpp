#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//image = cv2.imread('1.jpg')
//gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
//blur = cv2.GaussianBlur(gray, (5,5), 0)
//thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
//
//cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
//cnts = cnts[0] if len(cnts) == 2 else cnts[1]
//for c in cnts:
//area = cv2.contourArea(c)
//if area > 10000:
//cv2.drawContours(image, [c], -1, (36,255,12), 3)
//
//cv2.imwrite('thresh.png', thresh)
//cv2.imwrite('image.png', image)
//cv2.waitKey()

const char BLACK = 0;
const char WHITE = static_cast<char>(255);

class Area
{
public:
    Area(const int x, const int y, const int width, const int height) : _leftTop(x, y)
    {
        _areaImg = cv::Mat::zeros(width, height, 0); // CV_8U
    }

private:
    cv::Mat _areaImg;
    cv::Point _leftTop;
};

//Area constructAreaFrom(int x, int y, const cv::Mat& binaryImg, std::vector<bool>& visited)
void constructAreaFrom(Area& area, int x, int y, const cv::Mat& binaryImg, std::vector<bool>& visited)
{
    const int imgW = binaryImg.cols;
    const int imgH = binaryImg.rows;

    if (x >= 0 && x < imgW && y >= 0 && y < imgH && !visited[y * imgW + x] && binaryImg.at<char>(y, x) == WHITE)
    {
        visited[y * imgW + x] = true;
        constructAreaFrom(area, x - 1, y, binaryImg, visited);
        constructAreaFrom(area, x + 1, y, binaryImg, visited);
        constructAreaFrom(area, x, y - 1, binaryImg, visited);
        constructAreaFrom(area, x, y + 1, binaryImg, visited);
    }
}

void findAreasInBinaryImg(const cv::Mat& binaryImg, std::vector<Area>& areas)
{
    const int imgW = binaryImg.cols;
    const int imgH = binaryImg.rows;
    std::vector<bool> visited(imgW * imgH, false);

    for (int x = 0; x < imgW; x++)
    {
        for (int y = 0; y < imgH; y++)
        {
            if (visited[y * imgW + x] || binaryImg.at<char>(y, x) == BLACK) {
                continue;
            }

            // now we have a white pixel not visited before
            Area newArea(x, y, 0, 0);
            constructAreaFrom(newArea, x, y, binaryImg, visited);
            areas.push_back(newArea);
        }
    }

//    std::cout << "total areas found: " << numAreas << "\n";
}

int main()
{
    cv::Mat img = cv::imread("/home/lucky/dev/6sem/practice/count-objects/test2.jpg");
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_RGB2GRAY);

    cv::Mat blurredImg;
    cv::GaussianBlur(grayImg, blurredImg, cv::Size(15, 15), 0);

    cv::Mat binaryImg;
    cv::threshold(blurredImg, binaryImg, 0, 255,
                  cv::THRESH_BINARY + cv::THRESH_OTSU);

//    std::cout << binaryImg.type() << "\n";

//    cv::findContours(binaryImg, )
    std::vector<Area> areas;
    findAreasInBinaryImg(binaryImg, areas);

    std::cout << "Areas found: " << areas.size() << "\n";

    cv::imshow("window", binaryImg);

    cv::waitKey();
    return 0;
}
