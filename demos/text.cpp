#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    cv::VideoCapture cap(1);

    cap.set(CV_CAP_PROP_FPS, 23);

    cv::namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);

    while (1) {
        cv::Mat frame;
        cap.read(frame);
        putText(frame, "Somos Dario y Dario haciendo nuestro PF", cv::Point(5,50), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,255,0), 2.0);
        cv::imshow("MyVideo", frame);

        if (cv::waitKey(30) == 27) {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

    return 0;
}
