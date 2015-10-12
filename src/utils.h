#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>


void printMat(std::string name, cv::Mat mat) {

    // Assert mat has type double
    assert(mat.type() == CV_64F);

    cv::Size size = mat.size();

    std::cout << name << " size: " << size.height << "x" << size.width << std::endl;

    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            std::cout << mat.at<double>(x, y) << " ";
        }
        std::cout << std::endl;
    }
}

#endif
