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


std::pair<pf::Corners, bool> findChessboardCorners(cv::Mat frame) {

    pf::Corners corners;

    /** 
    Se utilizan los flags CV_CALIB_CB_ADAPTIVE_THRESH y 
    CV_CALIB_CB_NORMALIZE_IMAGE los cuales de forma empírica 
    han traído los mejores resultados. 
    El primer flag indica que se usa una cota adaptativa para 
    convertir la imagen a blanco y negro en vez de usar una cota fija. 
    El segundo flag obliga a normalizar la imagen gamma antes 
    de aplicar la cota antes mencionada.
    */
    
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

        /**
        CV_TERMCRIT_EPS y CV_TERMCRIT_ITER le indica al algoritmo 
        que la terminación del mismo será luego de que los 
        resultados convergen a cierto valor o luego de determinada 
        cantidad de iteraciones. Los siguientes parámetros indican 
        los valores de terminación concretamente.
        */

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


std::vector<pf::Corners> getCornersSamples(size_t index) {

    cv::Size numSquares(NUM_HOR_SQUARES, NUM_VER_SQUARES);
    cv::VideoCapture capture(index + 1);

    if (!capture.isOpened()) {
        std::cerr << "Can't open the camera" << std::endl;
        std::exit(-1);
    }

    capture.set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);

    std::vector<pf::Corners> cornersSamples;
    bool started = false;
    clock_t time = 0;

    while (cornersSamples.size() < NUM_FRAMES) {

        // Capture frame
        cv::Mat frame;
        capture >> frame;

        // Find chessboard corners
        auto found = findChessboardCorners(frame);

        if (found.second && started && clock() - time > DELAY_BETWEEN_FRAMES) {
            time = clock();
            cornersSamples.push_back(found.first);
            cv::bitwise_not(frame, frame);
        }

        // Show image
        cv::drawChessboardCorners(frame, numSquares, cv::Mat(found.first), found.second);
        cv::imshow("Calibrate", frame);

        // Wait for 's' to start
        if (cv::waitKey(100) == 's') {
            started = true;
        }
    }

    return cornersSamples;
}


int main(int argc, char **argv) {

    // Get corner samples for both cameras
    std::vector<pf::Corners> corners_image_1 = getCornersSamples(CAMERA_1);
    std::vector<pf::Corners> corners_image_2 = getCornersSamples(CAMERA_2);

    // Corner positions in the board space
    std::vector<cv::Point3f> corners;

    for (int y = 0; y < NUM_VER_SQUARES; y++) {
        for (int x = 0; x < NUM_HOR_SQUARES; x++) {
            corners.push_back(cv::Point3f(y * SQUARE_SIZE, x * SQUARE_SIZE, 0));
        }
    }

    pf::StereoParameters params = pf::StereoParameters::fromCorners(corners, corners_image_1, corners_image_2);

    cv::FileStorage fs(argc == 2 ? argv[1] : PARAMS_FILE, cv::FileStorage::WRITE);
    fs << params;

    return 0;
}
