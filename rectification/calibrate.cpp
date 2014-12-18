#include <vector>
#include <cstdio>
#include <cstdarg>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#define NUM_HOR_SQUARES 7
#define NUM_VER_SQUARES 5
#define SQUARE_SIZE 2.75f
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define NUM_FRAMES 25
#define DELAY_BETWEEN_FRAMES (CLOCKS_PER_SEC * 0.1)

// Do not change
#define FRAMES_PER_SECOND 23

typedef std::vector<cv::Point2f> Corners;

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
    cv::VideoCapture capture(index);

    if (!capture.isOpened()) {
        error("Error opening the camera (index %zu)", index);
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
        cv::imshow("Image View", frame);

        // Wait for 's' to start
        if (cv::waitKey(100) == 's') {
            started = true;
        }
    }

    return cornersSamples;
}

std::pair<cv::Mat, cv::Mat> getIntrinsicParameters(std::vector<Corners> imagePoints) {

    cv::Size numSquares(NUM_HOR_SQUARES, NUM_VER_SQUARES);

    // Corner positions in the board space
    std::vector<cv::Point3f> corners;

    for (int y = 0; y < NUM_VER_SQUARES; y++) {
        for (int x = 0; x < NUM_HOR_SQUARES; x++) {
            corners.push_back(cv::Point3f(y * SQUARE_SIZE, x * SQUARE_SIZE, 0));
        }
    }

    // Calibrate
    std::vector<std::vector<cv::Point3f> > objectPoints(imagePoints.size(), corners);
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeff = cv::Mat::zeros(8, 1, CV_64F);

    double rms = cv::calibrateCamera(
        objectPoints,
        imagePoints,
        cv::Size(FRAME_WIDTH, FRAME_HEIGHT),
        cameraMatrix,
        distCoeff,
        rvecs,
        tvecs,
        CV_CALIB_RATIONAL_MODEL|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5
    );

    return std::make_pair(cameraMatrix, distCoeff);
}

int main(int argc, char **argv) {

    if (argc != 3) {
        error("Usage: %s CAMERA_INDEX OUTPUT_YML_FILE", argv[0]);
    }

    size_t index = std::atoi(argv[1]);

    std::vector<Corners> imagePoints = getCornersSamples(index);
    std::pair<cv::Mat, cv::Mat> parameters = getIntrinsicParameters(imagePoints);

    // Save intrinsic parameters
    cv::FileStorage fs(argv[2], cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        error("Error opening destination file");
    }

    fs << "Camera_Matrix" << parameters.first;
    fs << "Distortion_Coefficients" << parameters.second;
}
