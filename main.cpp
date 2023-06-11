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
        _areaImg = cv::Mat::zeros(height, width, 0); // CV_8U
    }

    void show(const std::string& title) const
    {
        cv::imshow(title, _areaImg);
        cv::waitKey();
    }

    int width() const { return _areaImg.cols; }
    int height() const { return _areaImg.rows; }

    cv::Point leftTop() const { return _leftTop; }

    int computeArea() const
    {
        // the number of pixels in the object. that means the number of white pixels.
        int area = 0;
        const int w = width();
        const int h = height();

        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                if (_areaImg.at<char>(y, x) == WHITE)
                {
                    area++;
                }
            }
        }

        return area;
    }

    int computePerimeter() const
    {
        int perimeter = 0;
        const int w = width();
        const int h = height();

        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                if (_areaImg.at<char>(y, x) == WHITE)
                {
                    if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
                    {
                        perimeter++;
                    }
                    else if (_areaImg.at<char>(y, x - 1) != WHITE ||
                            _areaImg.at<char>(y, x + 1) != WHITE ||
                            _areaImg.at<char>(y - 1, x) != WHITE ||
                            _areaImg.at<char>(y + 1, x) != WHITE)
                    {
                        perimeter++;
                    }
                }
            }
        }

        return perimeter;
    }

    void addPoint(const int x, const int y, const char color)
    {
        int xOnArea = x - _leftTop.x;
        int yOnArea = y - _leftTop.y;

        if (xOnArea < 0)
        {
            _leftTop.x += xOnArea;
            addColumnsLeft(abs(xOnArea));
            xOnArea = 0;
        }
        if (yOnArea < 0)
        {
            _leftTop.y += yOnArea;
            addRowsTop(abs(yOnArea));
            yOnArea = 0;
        }
        if (xOnArea >= _areaImg.cols)
        {
            const int colsToAdd = xOnArea - _areaImg.cols + 1;
            addColumnsRight(colsToAdd);
        }
        if (yOnArea >= _areaImg.rows)
        {
            const int rowsToAdd = yOnArea - _areaImg.rows + 1;
            addRowsBottom(rowsToAdd);
        }

        _areaImg.at<char>(yOnArea, xOnArea) = color;
    }

    void addColumnsLeft(int num)
    {
        // the column is a cv::Mat (width = 1, height = _areaImg.height)
        cv::Mat newColumns = cv::Mat::zeros(_areaImg.rows, num, 0);
        cv::hconcat(newColumns, _areaImg, _areaImg);
    }

    void addColumnsRight(int num)
    {
        cv::Mat newColumns = cv::Mat::zeros(_areaImg.rows, num, 0);
        cv::hconcat(_areaImg, newColumns, _areaImg);
    }

    void addRowsTop(int num)
    {
        cv::Mat newRows = cv::Mat::zeros(num, _areaImg.cols, 0);
        cv::vconcat(newRows, _areaImg, _areaImg);
    }

    void addRowsBottom(int num)
    {
        cv::Mat newRows = cv::Mat::zeros(num, _areaImg.cols, 0);
        cv::vconcat(_areaImg, newRows, _areaImg);
    }

private:
    cv::Mat _areaImg;
    cv::Point _leftTop;
};

void constructAreaFrom(Area& area, int x, int y, const cv::Mat& binaryImg, std::vector<bool>& visited)
{
    const int imgW = binaryImg.cols;
    const int imgH = binaryImg.rows;

    if (x >= 0 && x < imgW && y >= 0 && y < imgH && !visited[y * imgW + x] && binaryImg.at<char>(y, x) == WHITE)
    {
        // add this point to the area
        area.addPoint(x, y, binaryImg.at<char>(y, x));

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
            if (visited[y * imgW + x] || binaryImg.at<char>(y, x) == BLACK)
            {
                continue;
            }

            // now we have a white pixel not visited before
            Area newArea(x, y, 0, 0);
            constructAreaFrom(newArea, x, y, binaryImg, visited);

            areas.push_back(newArea);
        }
    }
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

    std::vector<Area> areas;
    findAreasInBinaryImg(binaryImg, areas);

    std::cout << "areas found: " << areas.size() << "\n";
    std::cout << "areas info: \n\n";
    for (const Area& a : areas)
    {
        std::cout << "area:\n";
        std::cout << "left top: " << a.leftTop().x << ", " << a.leftTop().y << "\n";
        std::cout << "width, height: " << a.width() << ", " << a.height() << "\n\n";
    }

    cv::imshow("window", binaryImg);

    cv::waitKey();
    return 0;
}
