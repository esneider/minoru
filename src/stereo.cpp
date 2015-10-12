#include "elas/elas.h"
#include "stereo.h"


pf::StereoParameters pf::StereoParameters::fromCorners(
    std::vector<cv::Point3f> corner_coords,
    std::vector<Corners> corners_image_1,
    std::vector<Corners> corners_image_2)
{
    std::vector<std::vector<cv::Point3f> > objectPoints(corners_image_1.size(), corner_coords);
    pf::StereoParameters params;

    // Calibrate
    double rms = cv::stereoCalibrate(
        objectPoints,
        corners_image_1,
        corners_image_2,
        params.cameraMatrix[CAMERA_1],
        params.distCoeffs[CAMERA_1],
        params.cameraMatrix[CAMERA_2],
        params.distCoeffs[CAMERA_2],
        params.size,
        params.R,
        params.T,
        params.E,
        params.F,
        cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 1e-6),
        CV_CALIB_RATIONAL_MODEL | CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_PRINCIPAL_POINT
    );

    std::cout << "RMS = " << rms << std::endl;

    // Rectify
    cv::stereoRectify(
        params.cameraMatrix[CAMERA_1],
        params.distCoeffs[CAMERA_1],
        params.cameraMatrix[CAMERA_2],
        params.distCoeffs[CAMERA_2],
        params.size,
        params.R,
        params.T,
        params.rotation[CAMERA_1],
        params.rotation[CAMERA_2],
        params.projection[CAMERA_1],
        params.projection[CAMERA_2],
        params.Q,
        CV_CALIB_ZERO_DISPARITY,
        -1,
        params.size
    );

    params.rotation[CAMERA_1] = cv::Mat::eye(3, 3, CV_64F);
    params.rotation[CAMERA_2] = cv::Mat::eye(3, 3, CV_64F);
    params.projection[CAMERA_1].at<double>(0, 2) = 0;
    params.projection[CAMERA_1].at<double>(1, 2) = 0;
    params.projection[CAMERA_2].at<double>(0, 2) = 0;
    params.projection[CAMERA_2].at<double>(1, 2) = 0;

    // Compute rectification maps
    for (int cam = 0; cam < 2; cam++) {
        cv::initUndistortRectifyMap(
            params.cameraMatrix[cam],
            params.distCoeffs[cam],
            params.rotation[cam],
            params.projection[cam],
            params.size,
            CV_32FC1,
            params.map[cam][0],
            params.map[cam][1]
        );
    }

    return params;
}


void pf::DisparityMap::displayHSV() {

    cv::Mat_<cv::Vec3b> depthImg(this->map.size());

    for (uint j = 0; j < (uint)this->map.cols; j++) {
        for (uint i = 0; i < (uint)this->map.rows; i++) {

            float depth = std::min(this->map.at<float>(i, j) * 0.01f, 1.0f);
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

    cv::imshow("Disparity Map", depthImg);
}


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

    sbm(capture.rectified[CAMERA_1], capture.rectified[CAMERA_2], disp);
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

    sgbm(capture.rectified[CAMERA_1], capture.rectified[CAMERA_2], disp);
    cv::normalize(disp, map, 0, 255, CV_MINMAX, CV_8U);
}


pf::ELAS::ELAS(pf::StereoCapture capture) {

    static Elas::parameters elasParams;
    static Elas elas(elasParams);

    int32_t dims[] = {FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH};
    cv::Mat_<float> disp1 = cv::Mat_<float>(FRAME_HEIGHT, FRAME_WIDTH);
    cv::Mat_<float> disp2 = cv::Mat_<float>(FRAME_HEIGHT, FRAME_WIDTH);

    elas.process(
        capture.rectified[CAMERA_1].data,
        capture.rectified[CAMERA_2].data,
        (float*)disp1.data,
        (float*)disp2.data,
        dims
    );

    cv::normalize(disp1, map, 0, 255, CV_MINMAX, CV_8U);
}
