#include <iostream>
#include <cstdarg>
#include <cstdio>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>


#define FRAMES_PER_SECOND 23


void error(const char *format, ...) {

    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    std::printf("\n");
    va_end(args);
    std::exit(-1);
}

class Camera {

    size_t index;
    size_t fps;
    size_t width;
    size_t height;

    cv::Mat distCoeff;
    cv::Mat camMatrix;

    cv::VideoCapture capture;

public:

    Camera(size_t index, std::string camFile): index(index) {

        // Read intrinsic camera parameters
        cv::FileStorage fs(camFile, cv::FileStorage::READ);

        fs["Camera_Matrix"] >> camMatrix;
        fs["Distortion_Coefficients"] >> distCoeff;

        // Open camera stream
        capture.open(index);
        if (!capture.isOpened()) {
            error("Error opening the camera (index %zu)", index);
        }

        capture.set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);

        fps    = capture.get(CV_CAP_PROP_FPS);
        width  = capture.get(CV_CAP_PROP_FRAME_WIDTH);
        height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

        std::printf("Successfully opened camera %zu in %zux%zux%zu\n",
                    index, width, height, fps);
    }

    bool getFrame(cv::Mat &frame) {
        return capture.read(frame);
    }

    void undistort(cv::Mat &src, cv::Mat &dst) {
        cv::undistort(src, dst, camMatrix, distCoeff);
    }
};

int main(int argc, char **argv) {

    if (argc != 3) {
        error("Usage: %s CAM_MATRIX_L CAM_MATRIX_R", argv[0]);
    }

    Camera  left(1, argv[1]);
    Camera right(2, argv[2]);

    cv::namedWindow("Camera1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Camera2", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Matches", CV_WINDOW_AUTOSIZE);

    bool undistort = false;
    double cotaY = 20;
    double cotaX = 20;
    bool video = true;
    bool flagrectify = false;
    bool nextFrame = false;
    std::vector<cv::Point2f> leftPoints;
    std::vector<cv::Point2f> rightPoints;

    cv::Mat frame1;
    cv::Mat frame2;

    while (1) {
        if (video || nextFrame) {
            cv::Mat frame1pre;
            cv::Mat frame2pre;
            bool success1 =  left.getFrame(frame1pre);
            bool success2 = right.getFrame(frame2pre);

            if (undistort) {
                 left.undistort(frame1pre, frame1);
                right.undistort(frame2pre, frame2);
            }
            else {
                frame1 = frame1pre;
                frame2 = frame2pre;
            }
            if (!success1 || !success2) {
                std::cout << "Cannot read a frame from video stream" << std::endl;
                break;
            }

            cv::Mat gray1;
            cv::Mat gray2;

            cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
            cv::cvtColor(frame2, gray2, CV_BGR2GRAY);

            std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
            cv::SiftFeatureDetector detector;

            detector.detect(gray1, keypoints_1);
            detector.detect(gray2, keypoints_2);

            cv::Mat descriptors_1, descriptors_2;
            cv::SiftDescriptorExtractor extractor;

            extractor.compute(gray1, keypoints_1, descriptors_1);
            extractor.compute(gray2, keypoints_2, descriptors_2);

            std::vector<cv::DMatch> matches;
            std::vector<cv::DMatch> good_matches;
            cv::FlannBasedMatcher matcher;

            matcher.match(descriptors_1, descriptors_2, matches);

            double max_dist = 0;
            double min_dist = 100;

            for(int i = 0; i < descriptors_1.rows; i++) {
                double dist = matches[i].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }

            //std::cout << matches[0] << std::endl;
            //First filter: distancia en Y acotada. Distancia en X
            for(int i = 0; i < matches.size(); i++) {

                double distX = abs(keypoints_1[matches[i].queryIdx].pt.x - keypoints_2[matches[i].trainIdx].pt.x);
                double distY = abs(keypoints_1[matches[i].queryIdx].pt.y - keypoints_2[matches[i].trainIdx].pt.y);

                //std::cout << "X" << std::endl;
                //std::cout << distX << std::endl;
                //std::cout << "Y" << std::endl;
                //std::cout << distY << std::endl;
                //if (matches[i].distance <= cv::max(2 * min_dist, 0.02)) {
                if (distY < cotaY && distX < cotaX) {
                    good_matches.push_back(matches[i]);
                }
            }

            for(int i = 0; i < good_matches.size(); i++) {
                leftPoints.push_back(keypoints_1[good_matches[i].queryIdx].pt);
                rightPoints.push_back(keypoints_2[good_matches[i].trainIdx].pt);
            }


            cv::Mat img_matches;
            cv::drawMatches(frame1, keypoints_1, frame2, keypoints_2,
                    good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

            cv::imshow("Matches", img_matches);

            nextFrame = false;
        }

        if (flagrectify) {
            cv::Mat fundMat;

            fundMat = cv::findFundamentalMat(leftPoints, rightPoints);

            cv::Mat h1;
            cv::Mat h2;

            cv::stereoRectifyUncalibrated(leftPoints, rightPoints, fundMat, frame1.size(), h1, h2, 3);

            cv::Mat hLeft;
            cv::Mat hRight;

            //hLeft = cameraMatrixL.inv() * h1 * cameraMatrixL;
            //hRight = cameraMatrixR.inv() * h2 * cameraMatrixR;

            //std::cout << hLeft << std::endl;
            //std::cout << hRight << std::endl;

            cv::Mat finalLeft;
            cv::Mat finalRight;

            cv::warpPerspective(frame1, finalLeft, h1, frame1.size());
            cv::warpPerspective(frame2, finalRight, h2, frame2.size());

            cv::imshow("Camera1", finalLeft);
            cv::imshow("Camera2", finalRight);
        }

        int key = cv::waitKey(30);
        if (key == 10) {
            std::cout << "Enter pressed" << std::endl;
            //imwrite("cameraL.png", frame1);
            //imwrite("cameraR.png", frame2);
            //imwrite("matches.png", img_matches);
        }
        if (key == 'u') {
            undistort = !undistort;
        }
        if (key == 'v') {
            video = !video;
        }
        if (key == 'r') {
            flagrectify = !flagrectify;
        }
        if (key == 'w') {
            cotaY++;
            std::cout << cotaY << std::endl;
        }
        if (key == 's') {
            cotaY--;
            std::cout << cotaY << std::endl;
        }
        if (key == 'a')
        {
            cotaX--;
            std::cout << cotaX << std::endl;
        }
        if (key == 'd') {
            cotaX++;
            std::cout << cotaX << std::endl;
        }
        if (key == 'n') {
            nextFrame = true;
        }
        if (key == 27) {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

    return 0;
}
