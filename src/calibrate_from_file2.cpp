#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "stereo.h"
#include "utils.h"


#define NUM_HOR_SQUARES 7
#define NUM_VER_SQUARES 5
#define SQUARE_SIZE 2.75f
#define NUM_FRAMES 17


std::pair<pf::Corners, bool> findChessboardCorners(cv::Mat frame) {

    pf::Corners corners;

    bool found = cv::findChessboardCorners(
        frame,
        cv::Size(NUM_HOR_SQUARES, NUM_VER_SQUARES),
        corners,
        CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE
    );

    // Improve coordinate accuracy
    if (found) {
        cv::Mat frameGray;
        cv::cvtColor(frame, frameGray, CV_BGR2GRAY);
        cv::cornerSubPix(
            frameGray,
            corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1)
        );
    }

    return make_pair(corners, found);
}


int main(int argc, char **argv) {

    const char *path = "../samples/calibration/test_%s_%u.png";
    pf::ImageCamera camera = pf::ImageCamera(path);

    // Get corner samples for both cameras
    std::vector<pf::Corners> corner_samples[2];
    cv::Mat img[2];

	int successes=0;

	std::vector<pf::Corners> cornervector_l;
	std::vector<pf::Corners> cornervector_r;

	std::vector<cv::Point3f> point3dvector;

    for (int y = 0; y < NUM_VER_SQUARES; y++) {
        for (int x = 0; x < NUM_HOR_SQUARES; x++) {
            point3dvector.push_back(cv::Point3f(y * SQUARE_SIZE, x * SQUARE_SIZE, 0));
        }
    }

	while(successes<NUM_FRAMES)
	{
		camera.capture(img);

		cv::Mat frame_l = img[0];
		
		std::cout << frame_l.size().width << "," << frame_l.size().height << std::endl;
		cv::Mat frame_r = img[1];
		cv::Mat frame_l_grey;
		cv::Mat frame_r_grey;
		pf::Corners corners_l;
		pf::Corners corners_r;

		bool patternfoundl = cv::findChessboardCorners( 
			frame_l, 
			cv::Size(NUM_HOR_SQUARES, NUM_VER_SQUARES), 
			corners_l, 
			CV_CALIB_CB_FILTER_QUADS + CV_CALIB_CB_ADAPTIVE_THRESH );
		

	    bool patternfoundr = cv::findChessboardCorners( frame_r, cv::Size(NUM_HOR_SQUARES, NUM_VER_SQUARES), corners_r, CV_CALIB_CB_FILTER_QUADS + CV_CALIB_CB_ADAPTIVE_THRESH );
	    
		if(patternfoundl && patternfoundr)
	    {
	        cvtColor(frame_l,frame_l_grey,CV_RGB2GRAY);
	        cvtColor(frame_r,frame_r_grey,CV_RGB2GRAY);

	        cornerSubPix(frame_l_grey,corners_l,cv::Size(6,6),cv::Size(-1,-1),cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,50000000000,0.0000000000001));
	        cornerSubPix(frame_r_grey,corners_r,cv::Size(6,6),cv::Size(-1,-1),cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,50000000000,0.0000000000001));
	    }

	    drawChessboardCorners( frame_l, cv::Size(NUM_HOR_SQUARES, NUM_VER_SQUARES), corners_l, patternfoundl );
	    drawChessboardCorners( frame_r, cv::Size(NUM_HOR_SQUARES, NUM_VER_SQUARES), corners_r, patternfoundr );
	
	    imshow( "Webcaml", frame_l );
	    imshow( "Webcamr", frame_r );
		//cv::waitKey();

	    if( corners_l.size() == (NUM_HOR_SQUARES*NUM_VER_SQUARES) && corners_r.size() == (NUM_HOR_SQUARES*NUM_VER_SQUARES) )
	    {
	        cornervector_l.push_back( corners_l );
	        cornervector_r.push_back( corners_r );
	        
	        successes++;
	    }
		
		
	}

	cv::destroyAllWindows();

	std::vector<std::vector<cv::Point3f> > objectPoints(cornervector_l.size(), point3dvector);
	
	cv::Mat intrinsics_l = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat intrinsics_r = cv::Mat::eye(3, 3, CV_64F);

	cv::Mat distortion_l = cv::Mat::zeros(8, 1, CV_64F);
	cv::Mat distortion_r = cv::Mat::zeros(8, 1, CV_64F);

	std::vector<cv::Mat> rvecs_l;
	std::vector<cv::Mat> tvecs_l;
	std::vector<cv::Mat> rvecs_r;
	std::vector<cv::Mat> tvecs_r;

	double rms_l = calibrateCamera( objectPoints, cornervector_l, cv::Size(640, 480),
		                    intrinsics_l, distortion_l, rvecs_l, tvecs_l, CV_CALIB_FIX_K3 , cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,150000000000000000,DBL_EPSILON ) );



	double rms_r = calibrateCamera( objectPoints, cornervector_r, cv::Size(640, 480),
		                    intrinsics_r, distortion_r, rvecs_r, tvecs_r, CV_CALIB_FIX_K3 , cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,150000000000000000,DBL_EPSILON ) );

	
	cv::Mat R;
	cv::Mat T;
	cv::Mat E;
	cv::Mat F;

	double rms_stereo = stereoCalibrate( objectPoints, cornervector_l, cornervector_r, intrinsics_l, distortion_l, intrinsics_r, distortion_r,
		                                cv::Size(640, 480), R, T, E, F, 
		                                cv::TermCriteria( cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 150000000000000000,DBL_EPSILON ), CV_CALIB_FIX_K3+CV_CALIB_FIX_INTRINSIC);

	printMat("R", R);
	printMat("T", T);

	Rodrigues(R, R);
	
	printMat("Rodri", R);

	std::cout << "RMS:" << rms_stereo << std::endl;

	cv::Mat rectify_l;
	cv::Mat rectify_r;
	cv::Mat projection_l;
	cv::Mat projection_r;
	cv::Mat Q;

	stereoRectify( intrinsics_l, distortion_l, intrinsics_r, distortion_r, cv::Size(640, 480), R, T,
		            rectify_l, rectify_r, projection_l, projection_r, Q);


    return 0;
}
