#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>

int main() {

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

    cv::namedWindow("MyVideo1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("MyVideo2", CV_WINDOW_AUTOSIZE);

    while (1) {
        cv::Mat frame1;
        cv::Mat frame2;
        bool success1 = cap1.read(frame1);
        bool success2 = cap2.read(frame2);

        if (!success1 || !success2) {
            std::cout << "Cannot read a frame from video stream" << std::endl;
            break;
        }

        cv::imshow("MyVideo1", frame1);
        cv::imshow("MyVideo2", frame2);

		int key = cv::waitKey(30);
		if (key == 10) {
			std::cout << "Enter pressed" << std::endl;
			imwrite("testL.png", frame1);
			imwrite("testR.png", frame2);
		}

        if (key == 27) {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

    return 0;
}
