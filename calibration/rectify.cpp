#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	const std::string inputFileLeft = argv[1];
	const std::string inputFileRight = argv[2];
	FileStorage fsRight(inputFileRight, FileStorage::READ);
	FileStorage fsLeft(inputFileLeft, FileStorage::READ);

	cv::Mat cameraMatrixL, cameraMatrixR, distCoeffL, distCoeffR;
	fsRight["Camera_Matrix"] >> cameraMatrixR;
	fsLeft["Camera_Matrix"] >> cameraMatrixL;
	fsRight["Distortion_Coefficients"] >> distCoeffR;
	fsLeft["Distortion_Coefficients"] >> distCoeffL;

	cout << "Camera Matrix Left" << cameraMatrixL << endl;
	cout << "Dist Coeff Left" << distCoeffL << endl;
	cout << "Camera Matrix Right" << cameraMatrixR << endl;
	cout << "Dist Coeff Right" << distCoeffR << endl;

	fsRight.release();
	fsLeft.release();

	cv::Mat_<double> R(3,3);             // 3x3 matrix, rotation left to right camera
	cv::Mat_<double> T(3,1);             // * 3 * x1 matrix, translation left to right proj. center

	R <<  0.87, -0.003, -0.46, 0.001, 0.999, -0.003, 0.46, 0.002, 0.89;
	T << 228, 0, 0;

	cv::Mat R1,R2,P1,P2,Q;   // you're safe to leave OutpuArrays empty !
	Size imgSize(640,480); // wild guess from you cameramat ( not that it matters )

	cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imgSize, R, T, R1, R2, P1, P2, Q);

	cerr << "Q" << Q << endl;
}
