#ifndef _STEREO_H_
#define _STEREO_H_

#include <vector>
#include <opencv2/opencv.hpp>


// StereoParameters file
#define PARAMS_FILE "params.yml"

// Camera properties
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define FRAMES_PER_SECOND 23

// Camera constants
#define CAMERA_1 0
#define CAMERA_2 1


namespace pf {

    typedef std::vector<cv::Point2f> Corners;
    typedef cv::Mat_<float> Map;
    typedef cv::Mat_<uint8_t> Image;

    class Camera {
        public:
            virtual void capture(cv::Mat (&img)[2]) = 0;
            virtual ~Camera() {}
    };

    class VideoCamera: public Camera {

        private:
            cv::VideoCapture caps[2];

        public:
            VideoCamera() {

                for (int cam = 0; cam < 2; cam++) {
                    caps[cam] = cv::VideoCapture(cam + 1);

                    if (!caps[cam].isOpened()) {
                        std::cerr << "Can't open camera " << (cam + 1) << std::endl;
                        return;
                    }

                    caps[cam].set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);
                }
            }

            void capture(cv::Mat (&img)[2]) {

                for (int cam = 0; cam < 2; cam++) {
                    caps[cam].read(img[cam]);
                }
            }
    };

    class ImageCamera: public Camera {

        private:
            unsigned index;
            const char *format;

        public:
            ImageCamera(const char *format) {
                this->format = format;
                this->index = 0;
            }

            void capture(cv::Mat (&img)[2]) {

                static const char *cameras[2] = {"left", "right"};
                char path[128];
                index++;

                for (int cam = 0; cam < 2; cam++) {
                    std::sprintf(path, format, cameras[cam], index);
                    img[cam] = cv::imread(path, CV_LOAD_IMAGE_COLOR);
                }
            }
    };

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

            static StereoParameters fromCorners(
                std::vector<cv::Point3f> corner_coords,
                std::vector<Corners> corners_image_1,
                std::vector<Corners> corners_image_2);

            static StereoParameters fromFile(std::string filename) {

                cv::FileStorage fs(filename, cv::FileStorage::READ);
                StereoParameters params;
                fs >> params;
                return params;
            }

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
            Image rectify(cv::Mat capture, cv::Mat map[2]) {

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
                    cv::imshow("Camera " + std::to_string(cam), captures[cam]);
                }
            }

            void displayRectified() {

                for (int cam = 0; cam < 2; cam++) {
                    cv::imshow("Camera " + std::to_string(cam), rectified[cam]);
                }
            }
    };

    class DisparityMap {
        public:
            StereoCapture capture;
            Image map;
            Map disparity;
            const std::string name;

            DisparityMap(StereoCapture capture, std::string name):
                capture(capture), name(name) {}
            virtual void compute() = 0;
            virtual ~DisparityMap() {}

            void displayMap() {
                cv::imshow("Disparity Map", map);
            }

            void displayHSV();
    };

    class BM: public DisparityMap {
        public:
            BM(StereoCapture capture): DisparityMap(capture, "BM") {}
            void compute();
    };

    class SGBM: public DisparityMap {
        public:
            SGBM(StereoCapture capture): DisparityMap(capture, "SGBM") {}
            void compute();
    };

    class ELAS: public DisparityMap {
        public:
            ELAS(StereoCapture capture): DisparityMap(capture, "ELAS") {}
            void compute();
    };

} // namespace pf

#endif
