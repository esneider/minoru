#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "stereo.h"


#define NUM_HOR_SQUARES 7
#define NUM_VER_SQUARES 5
#define SQUARE_SIZE 2.75f
#define NUM_FRAMES 17


std::pair<pf::Corners, bool> findChessboardCorners(cv::Mat frame) {

    pf::Corners corners;

    bool found = cv::findChessboardCorners(
        frame,
        cv::Size(NUM_HOR_SQUARES, NUM_VER_SQUARES),
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
    }

    return make_pair(corners, found);
}


int main(int argc, char **argv) {

    const char *path = "/home/dario/Desktop/Parallels Shared Folders/Dropbox/pf-minoru/good_calib/test_%s_%u.png";
    pf::ImageCamera camera = pf::ImageCamera(path);

    // Get corner samples for both cameras
    std::vector<pf::Corners> corner_samples[2];
    cv::Mat img[2];

    for (int sample = 0; sample < NUM_FRAMES; sample++) {

        camera.capture(img);

        auto found_1 = findChessboardCorners(img[0]);
        auto found_2 = findChessboardCorners(img[1]);

        if (found_1.second && found_2.second) {
            corner_samples[0].push_back(found_1.first);
            corner_samples[1].push_back(found_2.first);

            cv::Mat frame;
            cv::Size numSquares(NUM_HOR_SQUARES, NUM_VER_SQUARES);

            std::cout << "Sample " <<(sample + 1) << std::endl;

            cv::drawChessboardCorners(img[0], numSquares, cv::Mat(found_1.first), found_1.second);
            cv::imshow("Corners 1", img[0]);

            cv::drawChessboardCorners(img[1], numSquares, cv::Mat(found_2.first), found_2.second);
            cv::imshow("Corners 2", img[1]);

            cv::waitKey();
        }
    }

    std::cout << corner_samples[0].size() << std::endl;

    // Corner positions in the board space
    std::vector<cv::Point3f> corners;

    for (int y = 0; y < NUM_VER_SQUARES; y++) {
        for (int x = 0; x < NUM_HOR_SQUARES; x++) {
            corners.push_back(cv::Point3f(y * SQUARE_SIZE, x * SQUARE_SIZE, 0));
        }
    }

    pf::StereoParameters params = pf::StereoParameters::fromCorners(corners, corner_samples[0], corner_samples[1]);

    cv::FileStorage fs(argc == 2 ? argv[1] : PARAMS_FILE, cv::FileStorage::WRITE);
    fs << params;

    return 0;
}
