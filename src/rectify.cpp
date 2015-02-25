#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "elas/elas.h"
#include "stereo.h"


int main(int argc, char **argv) {

    cv::VideoCapture caps[2];

    caps[CAMERA_1] = cv::VideoCapture(CAMERA_1 + 1);
    caps[CAMERA_2] = cv::VideoCapture(CAMERA_2 + 1);

    caps[CAMERA_1].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);
    caps[CAMERA_2].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);

    cv::namedWindow("Camera0", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Camera1", CV_WINDOW_AUTOSIZE);

    Elas::parameters elasParams;
    Elas elas(elasParams);

    StereoParameters params;
    cv::FileStorage fs(argc == 2 ? argv[1] : PARAMS_FILE, cv::FileStorage::READ);
    fs >> params;

    while (true) {

        cv::Mat img[2], gimg[2], rimg[2];
        float disparity[2][FRAME_HEIGHT][FRAME_WIDTH];
        int32_t dims[] = {FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH};

        for (int cam = 0; cam < 2; cam++) {
            caps[cam].read(img[cam]);
            cv::cvtColor(img[cam], gimg[cam], CV_BGR2GRAY);
            cv::remap(gimg[cam], rimg[cam], params.map[cam][0], params.map[cam][1], cv::INTER_LINEAR);
            cv::imshow("Camera" + std::to_string(cam), rimg[cam]);
        }

        // matching function
        // inputs: pointers to left (I1) and right (I2) intensity image (uint8, input)
        //         pointers to left (D1) and right (D2) disparity image (float, output)
        //         dims[0] = width of I1 and I2
        //         dims[1] = height of I1 and I2
        //         dims[2] = bytes per line (often equal to width, but allowed to differ)
        //         note: D1 and D2 must be allocated before (bytes per line = width)
        //               if subsampling is not active their size is width x height,
        //               otherwise width/2 x height/2 (rounded towards zero)
        // void process (uint8_t* I1,uint8_t* I2,float* D1,float* D2,const int32_t* dims);

        elas.process(
            (uint8_t*)rimg[CAMERA_1].data,
            (uint8_t*)rimg[CAMERA_2].data,
            (float*)disparity[CAMERA_1],
            (float*)disparity[CAMERA_2],
            dims);

        char c = (char)cv::waitKey(30);
        if (c == 27) {
            break;
        }
    }
}
