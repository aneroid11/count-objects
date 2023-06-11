#include <iostream>

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

int main()
{
    cv::Mat img = cv::imread("/home/lucky/dev/6sem/practice/count-objects/test2.jpg");
    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_RGB2GRAY);

    cv::Mat blurredImg;
    cv::GaussianBlur(grayImg, blurredImg, cv::Size(15, 15), 0);

    cv::Mat binaryImg;
    cv::threshold(blurredImg, binaryImg, 0, 255,
                  cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    
//    cv::findContours(binaryImg, )

    cv::imshow("window", binaryImg);

    cv::waitKey();
    return 0;
}
