#include "elas/elas.h"
#include "stereo.h"


pf::StereoParameters pf::StereoParameters::fromCorners(
    std::vector<cv::Point3f> corner_coords,
    std::vector<Corners> corners_image_1,
    std::vector<Corners> corners_image_2)
{
    std::vector<std::vector<cv::Point3f> > objectPoints(corners_image_1.size(), corner_coords);
    pf::StereoParameters params;

    std::vector<cv::Mat> rvecs[2];
    std::vector<cv::Mat> tvecs[2];

    /**
    se utiliza como flag únicamente CV_CALIB_FIX_K3 
    lo cual indica que el parámetro de distorsión radial 
    K3 no es modificado durante la optimización.
    */

    double rms_1 = calibrateCamera(
        objectPoints,
        corners_image_1,
        params.size,
        params.cameraMatrix[CAMERA_1],
        params.distCoeffs[CAMERA_1],
        rvecs[CAMERA_1],
        tvecs[CAMERA_1],
        CV_CALIB_FIX_K3,
        cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, INT_MAX, DBL_EPSILON)
    );

    double rms_2 = calibrateCamera(
        objectPoints,
        corners_image_2,
        params.size,
        params.cameraMatrix[CAMERA_2],
        params.distCoeffs[CAMERA_2],
        rvecs[CAMERA_2],
        tvecs[CAMERA_2],
        CV_CALIB_FIX_K3,
        cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, INT_MAX, DBL_EPSILON)
    );

    /**
    se agrega el flag CV_CALIB_FIX_INTRINSIC. 
    Esto indica que las matrices de las cámaras y los c
    oeficientes de distorsión son optimizados de modo tal que 
    las matrices R, T, E, y F sean estimadas.
    */

    // Calibrate
    double rms = cv::stereoCalibrate(
        objectPoints,
        corners_image_1,
        corners_image_2,
        params.cameraMatrix[CAMERA_1],
        params.distCoeffs[CAMERA_1],
        params.cameraMatrix[CAMERA_2],
        params.distCoeffs[CAMERA_2],
        params.size,
        params.R,
        params.T,
        params.E,
        params.F,
        cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,  INT_MAX, DBL_EPSILON),
        CV_CALIB_FIX_K3 | CV_CALIB_FIX_INTRINSIC
    );

    std::cout << "RMS = " << rms << std::endl;

    /**
    V_CALIB_ZERO_DISPARITY el cual hace que los puntos principales 
    de cada cámara tengan las mismas coordenadas en las vistas 
    rectificadas
    */

    // Rectify
    cv::stereoRectify(
        params.cameraMatrix[CAMERA_1],
        params.distCoeffs[CAMERA_1],
        params.cameraMatrix[CAMERA_2],
        params.distCoeffs[CAMERA_2],
        params.size,
        params.R,
        params.T,
        params.rotation[CAMERA_1],
        params.rotation[CAMERA_2],
        params.projection[CAMERA_1],
        params.projection[CAMERA_2],
        params.Q,
        CV_CALIB_ZERO_DISPARITY,
        -1,
        params.size
    );

    // Compute rectification maps
    for (int cam = 0; cam < 2; cam++) {
        cv::initUndistortRectifyMap(
            params.cameraMatrix[cam],
            params.distCoeffs[cam],
            params.rotation[cam],
            params.projection[cam],
            params.size,
            CV_32FC1,
            params.map[cam][0],
            params.map[cam][1]
        );
    }

    return params;
}


void pf::DisparityMap::displayHSV() {

    cv::Mat_<cv::Vec3b> depthImg(this->map.size());

    for (uint j = 0; j < (uint)this->map.cols; j++) {
        for (uint i = 0; i < (uint)this->map.rows; i++) {

            float depth = std::min(this->map.at<float>(i, j) * 0.01f, 1.0f);
            float h2 = 6.0f * (1.0f - depth);
            uint8_t x = (1.0f - std::fabs(std::fmod(h2, 2.0f) - 1.0f)) * 256;

            cv::Vec3b v;

            if (depth <= 0)  { v[0] = 0;   v[1] = 0;   v[2] = 0;   }
            else if (h2 < 1) { v[0] = 255; v[1] = x;   v[2] = 0;   }
            else if (h2 < 2) { v[0] = x;   v[1] = 255; v[2] = 0;   }
            else if (h2 < 3) { v[0] = 0;   v[1] = 255; v[2] = x;   }
            else if (h2 < 4) { v[0] = 0;   v[1] = x;   v[2] = 255; }
            else if (h2 < 5) { v[0] = x;   v[1] = 0;   v[2] = 255; }
            else             { v[0] = 255; v[1] = 0;   v[2] = x;   }

            depthImg.at<cv::Vec3b>(i, j) = v;
        }
    }

    cv::imshow("Disparity Map", depthImg);
}

/** Parametros de los algoritmos

sgbm.SADWindowSize = 5;

Esto indica el tamaño de las ventanas que serán comparadas entre ambas imágenes.

sgbm.numberOfDisparities = 192;

Esto indica el rango de disparidades buscadas se encuentra en [minDisparity, minDisparity+numberOfDisparities].

sgbm.preFilterCap = 4;

Se define una cota para los valores de salida a [-preFilterCap, preFilterCap]

sgbm.minDisparity = -64;

Se define el menor valor de disparidad que es tomado en cuenta.

sgbm.uniquenessRatio = 1;

Este valor define otro filtro para los valores aceptados por las disparidades.

sgbm.speckleWindowSize = 150;

Este valor es el tamaño máximo de regiones de disparidad para considerar (e invalidar) ruido Speckle.

sgbm.speckleRange = 2;

Esto define la máxima variación de las disparidades entre cada componente conectado.

sgbm.disp12MaxDiff = 10;

Bajo este valor se marca la diferencia permitida (en cantidad de píxeles) al hacer el chequeo izquierdo-derecho de disparidad.

sgbm.fullDP = false;

Al indicar este parámetro como falso, se utiliza una versión menos optimizada del algoritmo, mucho más económico en el uso de la memoria.

sgbm.P1 = 600;
sgbm.P2 = 2400;

Estos parámetros indican la suavidad de la disparidad. Cuanto más grandes son estos valores, mejores son los resultados. Ambos valores aplican castigos a la diferencia de disparidad entre píxeles vecinos.

sbm.state->preFilterSize = 5;

Se indica el tamaño de la ventana del filtro previo que se aplica.

sbm.state->textureThreshold = 507;

Se calcula la disparidad en donde la textura supera la cota indicada.

*/


pf::BM::BM(pf::StereoCapture capture) {

    std::cout << "Method: BM" << std::endl;

    cv::Mat disp;
    cv::StereoBM sbm;

    sbm.state->SADWindowSize = 9;
    sbm.state->numberOfDisparities = 112;
    sbm.state->preFilterSize = 5;
    sbm.state->preFilterCap = 61;
    sbm.state->minDisparity = -39;
    sbm.state->textureThreshold = 507;
    sbm.state->uniquenessRatio = 0;
    sbm.state->speckleWindowSize = 0;
    sbm.state->speckleRange = 8;
    sbm.state->disp12MaxDiff = 1;

    sbm(capture.rectified[CAMERA_1], capture.rectified[CAMERA_2], disp);
    disp.convertTo(disparity, CV_32F);
    cv::normalize(disp, map, 0, 255, CV_MINMAX, CV_8U);
}


pf::SGBM::SGBM(pf::StereoCapture capture) {

    std::cout << "Method: SGBM" << std::endl;

    cv::Mat disp;
    cv::StereoSGBM sgbm;

    sgbm.SADWindowSize = 5;
    sgbm.numberOfDisparities = 192;
    sgbm.preFilterCap = 4;
    sgbm.minDisparity = -64;
    sgbm.uniquenessRatio = 1;
    sgbm.speckleWindowSize = 150;
    sgbm.speckleRange = 2;
    sgbm.disp12MaxDiff = 10;
    sgbm.fullDP = false;
    sgbm.P1 = 600;
    sgbm.P2 = 2400;

    sgbm(capture.rectified[CAMERA_1], capture.rectified[CAMERA_2], disp);
    disp.convertTo(disparity, CV_32F);
    cv::normalize(disp, map, 0, 255, CV_MINMAX, CV_8U);
}


pf::ELAS::ELAS(pf::StereoCapture capture) {

    std::cout << "Method: ELAS" << std::endl;

    static Elas::parameters elasParams;
    static Elas elas(elasParams);

    int32_t dims[] = {FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH};
    cv::Mat_<float> disp1 = cv::Mat_<float>(FRAME_HEIGHT, FRAME_WIDTH);
    cv::Mat_<float> disp2 = cv::Mat_<float>(FRAME_HEIGHT, FRAME_WIDTH);

    elas.process(
        capture.rectified[CAMERA_1].data,
        capture.rectified[CAMERA_2].data,
        (float*)disp1.data,
        (float*)disp2.data,
        dims
    );

    disp1.convertTo(disparity, CV_32F);
    cv::normalize(disp1, map, 0, 255, CV_MINMAX, CV_8U);
}
