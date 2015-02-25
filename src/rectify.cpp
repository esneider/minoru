#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "elas/elas.h"
#include "stereo.h"


//function to show disparity map with HSV color

void printHSV(cv::Mat_<float>& disparityData, const char* windowName) {
  cv::Mat_<cv::Vec3b> disparity_data_color(disparityData.size());
  for (uint j = 0; j < (uint)disparityData.cols; j++) {
    for (uint i = 0; i < (uint)disparityData.rows; i++) {
      cv::Vec3b v;

      float val = std::min(disparityData.at<float>(i,j) * 0.01f, 1.0f);
      if (val <= 0) {
        v[0] = v[1] = v[2] = 0;
      } else {
        float h2 = 6.0f * (1.0f - val);
        unsigned char x  = (unsigned char)((1.0f - fabs(fmod(h2, 2.0f) - 1.0f))*255);
        if (0 <= h2&&h2<1) { v[0] = 255; v[1] = x; v[2] = 0; }
        else if (1 <= h2 && h2 < 2)  { v[0] = x; v[1] = 255; v[2] = 0; }
        else if (2 <= h2 && h2 < 3)  { v[0] = 0; v[1] = 255; v[2] = x; }
        else if (3 <= h2 && h2 < 4)  { v[0] = 0; v[1] = x; v[2] = 255; }
        else if (4 <= h2 && h2 < 5)  { v[0] = x; v[1] = 0; v[2] = 255; }
        else if (5 <= h2 && h2 <= 6) { v[0] = 255; v[1] = 0; v[2] = x; }
      }

      disparity_data_color.at<cv::Vec3b>(i, j) = v;
    }
  }

  // Create Window
  cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
  cv::imshow(windowName, disparity_data_color);
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

    Elas::parameters elasParams;
    Elas elas(elasParams);

    int32_t dims[] = {FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH};
    cv::Mat_<float> disparity[2] = {
        cv::Mat_<float>(FRAME_WIDTH, FRAME_HEIGHT),
        cv::Mat_<float>(FRAME_WIDTH, FRAME_HEIGHT)
    };

    cv::Mat img[2];
    cv::Mat_<uint8_t> gimg[2];
    cv::Mat_<uint8_t> rimg[2];

    while (true) {

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
            rimg[CAMERA_1].data,
            rimg[CAMERA_2].data,
            (float*)disparity[CAMERA_1].data,
            (float*)disparity[CAMERA_2].data,
            dims);
        
		printHSV(disparity[CAMERA_1], "Disparity Right Camera");
        printHSV(disparity[CAMERA_2], "Disparity Left Camera");

        char c = (char)cv::waitKey(30);
        if (c == 27) {
            break;
        }
    }
}
