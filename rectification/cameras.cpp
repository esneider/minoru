#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "opencv2/nonfree/features2d.hpp"



int main() {
    // cv::initModule_nonfree();
    cv::VideoCapture cap1(1);
    cv::VideoCapture cap2(2);

    if (!cap1.isOpened() || !cap2.isOpened()) {
        std::cout << "Cannot open the video cam" << std::endl;
        return -1;
    }

    cap1.set(CV_CAP_PROP_FPS, 23);
    cap2.set(CV_CAP_PROP_FPS, 23);

    double dWidth = cap2.get(CV_CAP_PROP_FRAME_WIDTH);
    double dHeight = cap2.get(CV_CAP_PROP_FRAME_HEIGHT);

    std::cout << "Frame size : " << dWidth << " x " << dHeight << std::endl;

    cv::namedWindow("Camera1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Camera2", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Matches", CV_WINDOW_AUTOSIZE);

    while (1) {
        cv::Mat frame1;
        cv::Mat frame2;

        bool success1 = cap1.read(frame1);
        bool success2 = cap2.read(frame2);

        if (!success1 || !success2) {
            std::cout << "Cannot read a frame from video stream" << std::endl;
            break;
        }

        cv::Mat gray1;
        cv::Mat gray2;

        cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
        cv::cvtColor(frame2, gray2, CV_BGR2GRAY);

        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::SurfFeatureDetector detector;

        detector.detect(gray1, keypoints_1);
        detector.detect(gray2, keypoints_2);

        cv::Mat descriptors_1, descriptors_2;
        cv::SurfDescriptorExtractor extractor;

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

        for(int i = 0; i < descriptors_1.rows; i++) {
            if (matches[i].distance <= cv::max(2 * min_dist, 0.02)) {
                good_matches.push_back(matches[i]);
            }
        }

        cv::Mat img_matches;
        cv::drawMatches(frame1, keypoints_1, frame2, keypoints_2,
                good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        cv::imshow("Matches", img_matches);
        // cv::imshow("Camera1", frame1);
        // cv::imshow("Camera2", frame2);

		int key = cv::waitKey(30);
		if (key == 10) {
			std::cout << "Enter pressed" << std::endl;
			imwrite("cameraL.png", frame1);
			imwrite("cameraR.png", frame2);
			imwrite("matches.png", img_matches);
		}

        if (key == 27) {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

    return 0;
}
