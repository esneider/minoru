#include <vector>
#include <opencv2/core/core.hpp>

#ifndef _STEREO_H_
#define _STEREO_H_


#define PARAMS_FILE "params.yml"

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define FRAMES_PER_SECOND 23

#define CAMERA_1 0
#define CAMERA_2 1


class StereoParameters {

public:
    // Intrinsic parameters
    std::vector<cv::Mat> cameraMatrix;
    std::vector<cv::Mat> distCoeffs;
    cv::Mat R;
    cv::Mat T;
    cv::Mat E;
    cv::Mat F;

    // Extrinsic parameters
    cv::Mat rotation[2];
    cv::Mat projection[2];
    cv::Mat Q;

    // Rectification map
    cv::Mat map[2][2];

    cv::Size size;

    StereoParameters():
        cameraMatrix(2, cv::Mat::eye(3, 3, CV_64F)),
        distCoeffs(2, cv::Mat::zeros(8, 1, CV_64F)),
        size(FRAME_WIDTH, FRAME_HEIGHT)
    {}

    friend cv::FileStorage &operator<<(cv::FileStorage &fs, const StereoParameters &params) {

        fs << "map00" << params.map[0][0];
        fs << "map01" << params.map[0][1];
        fs << "map10" << params.map[1][0];
        fs << "map11" << params.map[1][1];
        fs << "width" << params.size.width;
        fs << "height" << params.size.height;

        return fs;
    }

    friend cv::FileStorage &operator>>(cv::FileStorage &fs, StereoParameters &params) {

        fs["map00"] >> params.map[0][0];
        fs["map01"] >> params.map[0][1];
        fs["map10"] >> params.map[1][0];
        fs["map11"] >> params.map[1][1];
        fs["width"] >> params.size.width;
        fs["height"] >> params.size.height;

        return fs;
    }
};


#endif
