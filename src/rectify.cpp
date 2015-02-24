#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "stereo.h"


int main(int argc, char **argv) {

    StereoParameters params;
    cv::FileStorage fs(argc == 2 ? argv[1] : PARAMS_FILE, cv::FileStorage::READ);
    fs >> params;

    // cv::Mat canvas;
    // double sf;
    // int w, h;
    // sf = 600./MAX(params.size.width, params.size.height);
    // w = (int)(params.size.width*sf);
    // h = (int)(params.size.height*sf);
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
            cv::remap(img, rimg, params.map[cam][0], params.map[cam][1], cv::INTER_LINEAR);
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
}
