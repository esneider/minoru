#include <vector>
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define NUM_HOR_SQUARES 7
#define NUM_VER_SQUARES 5
#define SQUARE_SIZE 2.75f
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define NUM_FRAMES 25
#define DELAY_BETWEEN_FRAMES (CLOCKS_PER_SEC * 0.1)

// Do not change
#define FRAMES_PER_SECOND 23
#define CAMERA_1 0
#define CAMERA_2 1

typedef std::vector<cv::Point2f> Corners;

class StereoParameters {

public:
    // Intirinsic parameters
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
        size(FRAME_WIDTH, FRAME_HEIGHT) {

    }
};

void error(const char *format, ...) {

    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    std::printf("\n");
    va_end(args);
    std::exit(-1);
}

std::vector<Corners> getCornersSamples(size_t index) {

    cv::Size numSquares(NUM_HOR_SQUARES, NUM_VER_SQUARES);
    cv::VideoCapture capture(index + 1);

    if (!capture.isOpened()) {
        error("Error opening the camera (index %zu)", index + 1);
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
            CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE
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

StereoParameters *getParameters(std::vector<Corners> imagePoints1, std::vector<Corners> imagePoints2) {

    StereoParameters *params = new StereoParameters();

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
        objectPoints, imagePoints1, imagePoints2,
        params->cameraMatrix[CAMERA_1], params->distCoeffs[CAMERA_1],
        params->cameraMatrix[CAMERA_2], params->distCoeffs[CAMERA_2],
        params->size, params->R, params->T, params->E, params->F,
        cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 1e-6),
        CV_CALIB_RATIONAL_MODEL | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5
    );

    std::cout << "RMS = " << rms << std::endl;

    // Rectify
    cv::stereoRectify(
        params->cameraMatrix[CAMERA_1], params->distCoeffs[CAMERA_1],
        params->cameraMatrix[CAMERA_2], params->distCoeffs[CAMERA_2],
        params->size, params->R, params->T,
        params->rotation[CAMERA_1], params->rotation[CAMERA_2],
        params->projection[CAMERA_1], params->projection[CAMERA_2],
        params->Q,
        CV_CALIB_ZERO_DISPARITY,
        -1,
        params->size
    );

    // Compute rectification maps
    for (int cam = 0; cam < 2; cam++) {
        cv::initUndistortRectifyMap(
            params->cameraMatrix[cam], params->distCoeffs[cam],
            params->rotation[cam], params->projection[cam],
            params->size,
            CV_32FC1,
            params->map[cam][0], params->map[cam][1]
        );
    }

    return params;
}

int main(int argc, char **argv) {

    std::vector<Corners> imagePoints1 = getCornersSamples(CAMERA_1);
    std::vector<Corners> imagePoints2 = getCornersSamples(CAMERA_2);

    StereoParameters *params = getParameters(imagePoints1, imagePoints2);

    if (argc == 2) {
        cv::FileStorage fs(argv[1], cv::FileStorage::WRITE);

        if (!fs.isOpened()) {
            error("Error opening destination file");
        }

        fs << "Map00" << params->map[0][0];
        fs << "Map01" << params->map[0][1];
        fs << "Map10" << params->map[1][0];
        fs << "Map11" << params->map[1][1];
    }

    // cv::Mat canvas;
    // double sf;
    // int w, h;
    // sf = 600./MAX(params->size.width, params->size.height);
    // w = (int)(params->size.width*sf);
    // h = (int)(params->size.height*sf);
    // canvas.create(h, w*2, CV_8UC3);

    cv::VideoCapture caps[2];

    caps[CAMERA_1] = cv::VideoCapture(CAMERA_1 + 1);
    caps[CAMERA_2] = cv::VideoCapture(CAMERA_2 + 1);

    caps[CAMERA_1].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);
    caps[CAMERA_2].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);

    cv::namedWindow("Camera0", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Camera1", CV_WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat img, rimg;

        for (int cam = 0; cam < 2; cam++) {
            caps[cam].read(img);
            cv::remap(img, rimg, params->map[cam][0], params->map[cam][1], cv::INTER_LINEAR);
            cv::imshow("Camera" + std::to_string(cam), rimg);
            // cv::Mat canvasPart = canvas(cv::Rect(w*cam, 0, w, h));
            // cv::resize(rimg, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);
            // if (useCalibrated) {
            //     Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
            //               cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
            //     rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            // }
        }

        // for (int j = 0; j < canvas.rows; j += 16) {
        //     cv::line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
        // }

        // cv::imshow("rectified", canvas);

        char c = (char)cv::waitKey(30);
        if (c == 27 || c == 'q' || c == 'Q') {
            break;
        }
    }

    delete params;
}
