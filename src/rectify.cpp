#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "elas/elas.h"
#include "stereo.h"
#include "camera.h"
#include "disparity_map.h"


void printHSV(cv::Mat_<float> &disparity, const char *window) {

    cv::Mat_<cv::Vec3b> depthImg(disparity.size());

    for (uint j = 0; j < (uint)disparity.cols; j++) {
        for (uint i = 0; i < (uint)disparity.rows; i++) {

            float depth = std::min(disparity.at<float>(i, j) * 0.01f, 1.0f);
            float h2 = 6.0f * (1.0f - depth);
            uint8_t x = (1.0f - std::fabs(std::fmod(h2, 2.0f) - 1.0f)) * 256;

            cv::Vec3b v;

            if (depth <= 0)  { v[0] = 0;   v[1] = 0;   v[2] = 0;   }
            else if (h2 < 1) { v[0] = 255; v[1] = x;   v[2] = 0;   }
            else if (h2 < 2) { v[0] = x;   v[1] = 255; v[2] = 0;   }
            else if (h2 < 3) { v[0] = 0;   v[1] = 255; v[2] = x;   }
            else if (h2 < 4) { v[0] = 0;   v[1] = x;   v[2] = 255; }
            else if (h2 < 5) { v[0] = x;   v[1] = 0;   v[2] = 255; }
            else             { v[0] = 255; v[1] = 0;   v[2] = x;   }

            depthImg.at<cv::Vec3b>(i, j) = v;
        }
    }

    cv::namedWindow(window, CV_WINDOW_AUTOSIZE);
    cv::imshow(window, depthImg);
}


int main(int argc, char **argv) {

    cv::VideoCapture caps[2];

    caps[CAMERA_1] = cv::VideoCapture(CAMERA_1 + 1);
    caps[CAMERA_2] = cv::VideoCapture(CAMERA_2 + 1);

    caps[CAMERA_1].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);
    caps[CAMERA_2].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);

    cv::namedWindow("Camera0", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Camera1", CV_WINDOW_AUTOSIZE);

    StereoParameters params;
    cv::FileStorage fs(argc == 2 ? argv[1] : PARAMS_FILE, cv::FileStorage::READ);
    fs >> params;

    cv::Mat img[2];
    pf::Image gimg[2];
    pf::Image rimg[2];
    pf::Image disparity;
    pf::DisparityMap *dm;

    char method = 's';

    while (true) {

        for (int cam = 0; cam < 2; cam++) {
            caps[cam].read(img[cam]);
        }

        pf::StereoCapture stereo(img, params.map);
        stereo.displayRectified();

        if (method == 'a') dm = new pf::BM(stereo);
        if (method == 's') dm = new pf::SGBM(stereo);
        if (method == 'd') dm = new pf::ELAS(stereo);

        dm->displayMap();
        // printHSV(dm->map, "Disparity Map");
        delete dm;


        char c = (char)cv::waitKey(30);
        if (c == 27) break;
        if (c == 'a' || c == 's' || c == 'd') method = c;
    }
}
