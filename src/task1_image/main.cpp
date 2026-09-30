#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;

int main() {
    Mat img = imread("resources/test_image.jpg");
    if (img.empty()) {
        std::cerr << "Cannot read image!\n";
        return  -1;
    }
    imshow("Original", img);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    imshow("Gray", gray);
    imwrite("result/task1_images/gray.png", gray);
    
    Mat equalized;
    equalizeHist(gray, equalized);
    imshow("Equalized", equalized);

    Mat meanImg, gaussianImg, medianImg;
    blur(gray, meanImg, Size(5,5));
    GaussianBlur(gray, gaussianImg, Size(5,5), 1.5);
    medianBlur(gray, medianImg, 5);
    imshow("Mean_filter", meanImg);
    imshow("Gaussian_filter", gaussianImg);
    imshow("Median_filter", medianImg);
    imwrite("result/task1_images/mean_filter.png", meanImg);
    imwrite("result/task1_images/gaussian_filter.png", gaussianImg);
    imwrite("result/task1_images/median_filter.png", medianImg);

    Mat binary,adaptive;
    threshold(gray, binary, 128, 255, THRESH_BINARY);
    adaptiveThreshold(gray, adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    imshow("Binary", binary);
    imshow("Adaptive", adaptive);
    Mat smooth, edges;
    GaussianBlur(gray, smooth, Size(5,5), 1.5);
    Canny(smooth, edges, 100, 200);
    imshow("Canny", edges);

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    std::vector<Mat> hsv_channels;
    split(hsv, hsv_channels);
    Mat H = hsv_channels[0];
    Mat S = hsv_channels[1];
    Mat V = hsv_channels[2];
    imshow("Hsv_h", H);
    imshow("Hsv_s", S);
    imshow("Hsv_v", V);
    imwrite("result/task1_images/hsv_h.png", H);
    imwrite("result/task1_images/hsv_s.png", S);
    imwrite("result/task1_images/hsv_v.png", V);

    Mat maskLow, maskHigh, mask;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), maskLow);
    inRange(hsv, Scalar(170, 100, 100), Scalar(179, 255, 255), maskHigh);
    bitwise_or(maskLow, maskHigh, mask);
    imshow("Red_mask", mask);
    imwrite("result/task1_images/red_mask.png", mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat eroded, dilated, opened, closed;
    dilate(mask, dilated, kernel);
    erode(mask, eroded, kernel);
    morphologyEx(mask, opened, MORPH_OPEN, kernel);
    morphologyEx(mask, closed, MORPH_CLOSE, kernel);
    imshow("Erode", eroded);
    imshow("Dilate", dilated);
    imshow("Open", opened);
    imshow("Close", closed);
    imwrite("result/task1_images/erode.png", eroded);
    imwrite("result/task1_images/dilate.png", dilated);
    imwrite("result/task1_images/open.png", opened);
    imwrite("result/task1_images/close.png", closed);

    Mat contourSource;
    morphologyEx(mask, contourSource, MORPH_OPEN, kernel); 
    morphologyEx(contourSource, contourSource, MORPH_CLOSE, kernel); 
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(contourSource, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat contourImg = Mat::zeros(img.size(), CV_8UC3);
    drawContours(contourImg, contours, -1, Scalar(0, 255, 0), 2);
    imshow("Contours", contourImg);
    Mat result = img.clone();
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area < 500.0) continue;
        Rect box = boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width) / box.height;
        if (ratio < 0.2 || ratio > 5.0) continue;
        rectangle(result, box, Scalar(0, 0, 255), 2);
        drawContours(result, contours, static_cast<int>(i),
        Scalar(0, 255, 0), 2);
        putText(result, std::to_string((int)area),
        Point(box.x, box.y - 5),
        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
    }
    imshow("Contours_boxes", result);
    imwrite("result/task1_images/contours_boxes.png", result);

    Mat drawing = img.clone();
    circle(drawing, Point(200, 200), 50, Scalar(255, 0, 0), 2);
    rectangle(drawing, Rect(200, 200, 100, 50), Scalar(0, 255, 0), 2);
    putText(drawing, "TASK1", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    imshow("Drawing", drawing);
    imwrite("result/task1_images/drawing.png", drawing);
    Point2f center(img.cols/2.0, img.rows/2.0);
    Mat rotMat = getRotationMatrix2D(center, 35, 1.0);
    Mat rotated;
    warpAffine(img, rotated, rotMat, img.size());
    imshow("Rotated_35deg", rotated);
    imwrite("result/task1_images/rotated_35deg.png", rotated);
    Rect roi(0,0,img.cols/2, img.rows/2);
    Mat crop = img(roi).clone();
    imshow("Crop_top_left", crop);
    imwrite("result/task1_images/crop_top_left.png", crop);
    waitKey(0);
    return 0;
}