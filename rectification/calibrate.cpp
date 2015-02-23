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
#define CAMERA_1 1
#define CAMERA_2 2

typedef std::vector<cv::Point2f> Corners;

class CameraParameters {

public:
    // Intirinsic parameters
	cv::Mat cameraMatrix1;
	cv::Mat distCoeff1;
	cv::Mat cameraMatrix2;
	cv::Mat distCoeff2;
	cv::Mat R;
	cv::Mat T;
	cv::Mat E;
	cv::Mat F;

    // Extrinsic parameters
	cv::Mat R1;
	cv::Mat R2;
	cv::Mat P1;
	cv::Mat P2;
	cv::Mat Q;

    // Transformation map
	cv::Mat rmap[2][2];

	cv::Size size;

    CameraParameters():
        cameraMatrix1(cv::Mat::eye(3, 3, CV_64F)),
        distCoeff1(cv::Mat::zeros(8, 1, CV_64F)),
        cameraMatrix2(cv::Mat::eye(3, 3, CV_64F)),
        distCoeff2(cv::Mat::zeros(8, 1, CV_64F)),
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

CameraParameters *getParameters(std::vector<Corners> imagePoints1, std::vector<Corners> imagePoints2) {

    cv::Size numSquares(NUM_HOR_SQUARES, NUM_VER_SQUARES);

    // Corner positions in the board space
    std::vector<cv::Point3f> corners;

    for (int y = 0; y < NUM_VER_SQUARES; y++) {
        for (int x = 0; x < NUM_HOR_SQUARES; x++) {
            corners.push_back(cv::Point3f(y * SQUARE_SIZE, x * SQUARE_SIZE, 0));
        }
    }

    // Calibrate
    std::vector<std::vector<cv::Point3f> > objectPoints(imagePoints1.size(), corners);
    CameraParameters *params = new CameraParameters();

	double rms = cv::stereoCalibrate(
		objectPoints,
		imagePoints1,
		imagePoints2,
		params->cameraMatrix1,
		params->distCoeff1,
		params->cameraMatrix2,
		params->distCoeff2,
		params->size,
		params->R,
		params->T,
		params->E,
		params->F
	);

	cv::stereoRectify(
		params->cameraMatrix1,
		params->distCoeff1,
		params->cameraMatrix2,
		params->distCoeff2,
		params->size,
		params->R,
		params->T,
		params->R1,
		params->R2,
		params->P1,
		params->P2,
		params->Q,
		CV_CALIB_ZERO_DISPARITY,
		-1,
		params->size
	);

	return params;
}

int main(int argc, char **argv) {

    std::vector<Corners> imagePoints1 = getCornersSamples(CAMERA_1);
	std::vector<Corners> imagePoints2 = getCornersSamples(CAMERA_2);

    CameraParameters *params = getParameters(imagePoints1, imagePoints2);

	cv::initUndistortRectifyMap(params->cameraMatrix1, params->distCoeff1, params->R1, params->P1, params->size, CV_16SC2, params->rmap[0][0], params->rmap[0][1]);

	cv::initUndistortRectifyMap(params->cameraMatrix2, params->distCoeff2, params->R2, params->P2, params->size, CV_16SC2, params->rmap[1][0], params->rmap[1][1]);


	cv::Mat canvas;
	double sf;
	int w, h;

	sf = 600./MAX(params->size.width, params->size.height);
	w = (int)(params->size.width*sf);
	h = (int)(params->size.height*sf);
	canvas.create(h, w*2, CV_8UC3);

	cv::VideoCapture caps[2];

	caps[0] = cv::VideoCapture(CAMERA_1);
	caps[1] = cv::VideoCapture(CAMERA_2);

	caps[0].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);
	caps[1].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);

	int k, j;

	while(1)
	{
		cv::Mat img, rimg, cimg;

		for( k = 0; k < 2; k++ ) {
			bool success = caps[k].read(img);
			cv::remap(img, rimg, params->rmap[k][0], params->rmap[k][1], cv::INTER_LINEAR);
			//cv::cvtColor(rimg, cimg, cv::COLOR_GRAY2BGR);
			cv::Mat canvasPart = canvas(cv::Rect(w*k, 0, w, h));
			cv::resize(rimg, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);

			/**
			if( useCalibrated ) {
				Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
						  cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
				rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
			}
			*/
		}

		for( j = 0; j < canvas.rows; j += 16 ) {
			cv::line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
		}

		cv::imshow("rectified", canvas);
		char c = (char)cv::waitKey(30);
		if( c == 27 || c == 'q' || c == 'Q' ) {
			break;
		}
	}

/*
    // Save intrinsic parameters
    cv::FileStorage fs(argv[2], cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        error("Error opening destination file");
    }

    fs << "Camera_Matrix" << params.first;
    fs << "Distortion_Coefficients" << params.second;
	*/
	delete params;
}
