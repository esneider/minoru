#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "stereo.h"


#define NUM_HOR_SQUARES 7
#define NUM_VER_SQUARES 5
#define SQUARE_SIZE 2.75f
#define NUM_FRAMES 15
#define DELAY_BETWEEN_FRAMES (CLOCKS_PER_SEC * 5)


typedef std::vector<cv::Point2f> Corners;


std::vector<Corners> getCornersSamples(size_t index) {

    cv::Size numSquares(NUM_HOR_SQUARES, NUM_VER_SQUARES);
    cv::VideoCapture capture(index + 1);

    if (!capture.isOpened()) {
        std::cerr << "Can't open the camera" << std::endl;
        std::exit(-1);
    }

    capture.set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);

    std::vector<Corners> cornersSamples;
    bool started = false;
    clock_t time = 0;

    while (cornersSamples.size() < NUM_FRAMES) {

        // Capture frame
        cv::Mat frame;
        capture >> frame;

        // Find chessboard corners
        Corners corners;

        bool found = cv::findChessboardCorners(
            frame,
            numSquares,
            corners,
            CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE
        );

        // Improve coordinate accuracy
        if (found) {
            cv::Mat frameGray;
            cv::cvtColor(frame, frameGray, CV_BGR2GRAY);
            cv::cornerSubPix(
                frameGray,
                corners,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1)
            );

            if (started && clock() - time > DELAY_BETWEEN_FRAMES) {
                time = clock();
                cornersSamples.push_back(corners);
                cv::bitwise_not(frame, frame);
            }
        }

        // Show image
        cv::drawChessboardCorners(frame, numSquares, cv::Mat(corners), found);
        cv::imshow("Calibrate", frame);

        // Wait for 's' to start
        if (cv::waitKey(100) == 's') {
            started = true;
        }
    }

    return cornersSamples;
}


pf::StereoParameters *getParameters(std::vector<Corners> imagePoints1, std::vector<Corners> imagePoints2) {

    pf::StereoParameters *params = new pf::StereoParameters();

    // Corner positions in the board space
    std::vector<cv::Point3f> corners;

    for (int y = 0; y < NUM_VER_SQUARES; y++) {
        for (int x = 0; x < NUM_HOR_SQUARES; x++) {
            corners.push_back(cv::Point3f(y * SQUARE_SIZE, x * SQUARE_SIZE, 0));
        }
    }

    std::vector<std::vector<cv::Point3f> > objectPoints(imagePoints1.size(), corners);

    // Calibrate
    double rms = cv::stereoCalibrate(
        objectPoints,
        imagePoints1,
        imagePoints2,
        params->cameraMatrix[CAMERA_1],
        params->distCoeffs[CAMERA_1],
        params->cameraMatrix[CAMERA_2],
        params->distCoeffs[CAMERA_2],
        params->size,
        params->R,
        params->T,
        params->E,
        params->F,
        cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 1e-6),
        CV_CALIB_RATIONAL_MODEL | CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_PRINCIPAL_POINT
    );

    std::cout << "RMS = " << rms << std::endl;

    // Rectify
    cv::stereoRectify(
        params->cameraMatrix[CAMERA_1],
        params->distCoeffs[CAMERA_1],
        params->cameraMatrix[CAMERA_2],
        params->distCoeffs[CAMERA_2],
        params->size,
        params->R,
        params->T,
        params->rotation[CAMERA_1],
        params->rotation[CAMERA_2],
        params->projection[CAMERA_1],
        params->projection[CAMERA_2],
        params->Q,
        CV_CALIB_ZERO_DISPARITY,
        -1,
        params->size
    );

    cv::Mat rot = params->rotation[CAMERA_1].t();
    cv::Mat trans = -params->projection[CAMERA_1];

    // Compute rectification maps
    for (int cam = 0; cam < 2; cam++) {
        cv::initUndistortRectifyMap(
            params->cameraMatrix[cam],
            params->distCoeffs[cam],
            params->rotation[cam] * rot,
            params->projection[cam] + trans,
            params->size,
            CV_32FC1,
            params->map[cam][0],
            params->map[cam][1]
        );
    }

    return params;
}


int main(int argc, char **argv) {

    std::vector<Corners> imagePoints1 = getCornersSamples(CAMERA_1);
    std::vector<Corners> imagePoints2 = getCornersSamples(CAMERA_2);

    pf::StereoParameters *params = getParameters(imagePoints1, imagePoints2);

    cv::FileStorage fs(argc == 2 ? argv[1] : PARAMS_FILE, cv::FileStorage::WRITE);
    fs << (*params);

    delete params;
    return 0;
}
