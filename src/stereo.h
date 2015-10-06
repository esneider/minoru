#ifndef _STEREO_H_
#define _STEREO_H_

#include <vector>
#include <opencv2/opencv.hpp>


#define PARAMS_FILE "params.yml"

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define FRAMES_PER_SECOND 23

#define CAMERA_1 0
#define CAMERA_2 1


namespace pf {

    typedef cv::Mat_<uint8_t> Image;

    class StereoParameters {

        public:
            // Intrinsic parameters
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
                size(FRAME_WIDTH, FRAME_HEIGHT)
            {}

            friend cv::FileStorage &operator<<(cv::FileStorage &fs, const StereoParameters &params) {

                fs << "map00" << params.map[0][0];
                fs << "map01" << params.map[0][1];
                fs << "map10" << params.map[1][0];
                fs << "map11" << params.map[1][1];
                fs << "width" << params.size.width;
                fs << "height" << params.size.height;

                return fs;
            }

            friend cv::FileStorage &operator>>(cv::FileStorage &fs, StereoParameters &params) {

                fs["map00"] >> params.map[0][0];
                fs["map01"] >> params.map[0][1];
                fs["map10"] >> params.map[1][0];
                fs["map11"] >> params.map[1][1];
                fs["width"] >> params.size.width;
                fs["height"] >> params.size.height;

                return fs;
            }
    };

    class StereoCapture {

        private:
            Image rectify(pf::Image capture, cv::Mat map[2]) {

                Image gray, rectified;
                cv::cvtColor(capture, gray, CV_BGR2GRAY);
                cv::remap(gray, rectified, map[0], map[1], cv::INTER_LINEAR);
                return rectified;
            }

        public:
            cv::Mat captures[2];
            Image rectified[2];

            StereoCapture(cv::Mat captures[2], cv::Mat map[2][2]) {

                for (int cam = 0; cam < 2; cam++) {
                    this->captures[cam] = captures[cam];
                    this->rectified[cam] = rectify(captures[cam], map[cam]);
                }
            }

            void displayCaptures() {

                for (int cam = 0; cam < 2; cam++) {
                    cv::imshow("Camera" + std::to_string(cam), captures[cam]);
                }
            }

            void displayRectified() {

                for (int cam = 0; cam < 2; cam++) {
                    cv::imshow("Camera" + std::to_string(cam), rectified[cam]);
                }
            }
    };

    class DisparityMap {
        public:
            Image map;

            virtual ~DisparityMap() {}

            void displayMap() {
                cv::imshow("Disparity Map", map);
            }
    };

    class BM: public DisparityMap {
        public:
            BM(StereoCapture capture);
    };

    class SGBM: public DisparityMap {
        public:
            SGBM(StereoCapture capture);
    };

    class ELAS: public DisparityMap {
        public:
            ELAS(StereoCapture capture);
    };

} // namespace pf


#endif
