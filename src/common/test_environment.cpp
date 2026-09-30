#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>

int main()
{
    std::cout << "C++ OK" << std::endl;

    cv::Mat img = cv::Mat::zeros(300, 300, CV_8UC3);

    Eigen::Matrix2d A;
    A << 1, 2,
         3, 4;

    std::cout << "Eigen OK" << std::endl;
    std::cout << A << std::endl;

    std::cout << "Ceres OK" << std::endl;

    cv::imshow("Test", img);
    cv::waitKey(0);

    return 0;
}