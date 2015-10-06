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

    if (argc != 2) {
        std::cout << "Usage: rectify_from_file img_cam_1 img_cam_2" << std::endl;
        return 1;
    }

    pf::StereoParameters params;
    cv::FileStorage fs(argc == 2 ? argv[1] : PARAMS_FILE, cv::FileStorage::READ);
    fs >> params;

    cv::Mat img[2];
    for (int cam = 0; cam < 2; cam++) {
        img[cam] = cv::imread(argv[cam + 1], CV_LOAD_IMAGE_COLOR);
    }

    pf::DisparityMap *dm;

    char method = 's';

    while (true) {

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
