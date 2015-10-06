#include "elas/elas.h"
#include "stereo.h"


pf::BM::BM(pf::StereoCapture capture) {

    cv::Mat disp;
    cv::StereoBM sbm;

    sbm.state->SADWindowSize = 9;
    sbm.state->numberOfDisparities = 112;
    sbm.state->preFilterSize = 5;
    sbm.state->preFilterCap = 61;
    sbm.state->minDisparity = -39;
    sbm.state->textureThreshold = 507;
    sbm.state->uniquenessRatio = 0;
    sbm.state->speckleWindowSize = 0;
    sbm.state->speckleRange = 8;
    sbm.state->disp12MaxDiff = 1;

    sbm(capture.captures[CAMERA_1], capture.captures[CAMERA_2], disp);
    cv::normalize(disp, map, 0, 255, CV_MINMAX, CV_8U);
}


pf::SGBM::SGBM(pf::StereoCapture capture) {

    cv::Mat disp;
    cv::StereoSGBM sgbm;

    sgbm.SADWindowSize = 5;
    sgbm.numberOfDisparities = 192;
    sgbm.preFilterCap = 4;
    sgbm.minDisparity = -64;
    sgbm.uniquenessRatio = 1;
    sgbm.speckleWindowSize = 150;
    sgbm.speckleRange = 2;
    sgbm.disp12MaxDiff = 10;
    sgbm.fullDP = false;
    sgbm.P1 = 600;
    sgbm.P2 = 2400;

    sgbm(capture.captures[CAMERA_1], capture.captures[CAMERA_2], disp);
    cv::normalize(disp, map, 0, 255, CV_MINMAX, CV_8U);
}


pf::ELAS::ELAS(pf::StereoCapture capture) {

    static Elas::parameters elasParams;
    static Elas elas(elasParams);

    int32_t dims[] = {FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH};
    cv::Mat_<float> disp1 = cv::Mat_<float>(FRAME_HEIGHT, FRAME_WIDTH);
    cv::Mat_<float> disp2 = cv::Mat_<float>(FRAME_HEIGHT, FRAME_WIDTH);

    elas.process(
        capture.captures[CAMERA_1].data,
        capture.captures[CAMERA_2].data,
        (float*)disp1.data,
        (float*)disp2.data,
        dims
    );

    cv::normalize(disp1, map, 0, 255, CV_MINMAX, CV_8U);
}

