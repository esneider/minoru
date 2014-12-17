#include <iostream>
#include <cstdarg>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

void error(const char *format, ...) {

    va_list args;
    va_start(args, format);
    vprintf(format, args);
    printf("\n");
    va_end(args);
    exit(-1);
}

class Camera {

    size_t index;
    size_t width;
    size_t height;
    size_t fps;

    cv::Mat distCoeff;
    cv::Mat cameraMatrix;
    cv::VideoCapture *capture;

public:

    Camera(size_t index, std::string cam_file): index(index) {

        cv::FileStorage fs(cam_file, cv::FileStorage::READ);

        fs["Camera_Matrix"] >> cameraMatrix;
        fs["Distortion_Coefficients"] >> distCoeff;

        capture = new cv::VideoCapture(index);
        if (!capture->isOpened()) {
            error("Error opening the camera (index %d)", index);
        }

        capture->set(CV_CAP_PROP_FPS, FRAMES_PER_SECOND);

        // capture->
    }
}

int main(int argc, char** argv) {

    if (argc != 2) {
        error("Usage: %s CAM_MATRIX_L CAM_MATRIX_R", argv[0]);
    }

    Camera  left(1, argv[1]);
    Camera right(2, argv[2]);

    cap1.set(CV_CAP_PROP_FPS, 23);
    cap2.set(CV_CAP_PROP_FPS, 23);

    double width  = cap2.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = cap2.get(CV_CAP_PROP_FRAME_HEIGHT);

    std::cout << "Frame size : " << width << " x " << height << std::endl;

    cv::namedWindow("Camera1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Camera2", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Matches", CV_WINDOW_AUTOSIZE);

    // Joda

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
            bool success1 = cap1.read(frame1pre);
            bool success2 = cap2.read(frame2pre);

            if (undistort) {
                cv::undistort(frame1pre, frame1, cameraMatrixL, distCoeffL);
                cv::undistort(frame2pre, frame2, cameraMatrixR, distCoeffR);
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
