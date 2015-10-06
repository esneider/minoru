#include <cstring>
#include <cassert>
#include "stereo.h"


struct Arguments {

    std::string paramsFile = PARAMS_FILE;
    std::string imageFiles[2];
    bool fromFile = false;

    Arguments(int argc, char **argv) {

        for (int arg = 0; arg < argc; arg++) {

            if (strcmp("--params", argv[arg]) == 0) {

                assert(++arg < argc);
                paramsFile = argv[arg];

            } else if (strcmp("--images", argv[arg]) == 0) {

                fromFile = true;

                for (int cam = 0; cam < 2; cam++) {
                    assert(++arg < argc);
                    imageFiles[cam] = argv[arg];
                }

            } else {
                std::cout << "Usage: rectify [--params file] [--images file_cam_1 file_cam_2]" << std::endl;
            }
        }
    }
};

int main(int argc, char **argv) {

    // Setup windows
    cv::namedWindow("Camera 0", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Camera 1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Disparity Map", CV_WINDOW_AUTOSIZE);

    // Parse command line arguments
    Arguments args(argc - 1, argv + 1);

    // Read camera parameters from file
    pf::StereoParameters params = pf::StereoParameters::fromFile(args.paramsFile);

    // Setup camera / read images from file
    pf::Camera *camera;
    cv::Mat img[2];

    if (args.fromFile) {
        for (int cam = 0; cam < 2; cam++) {
            img[cam] = cv::imread(args.imageFiles[cam], CV_LOAD_IMAGE_COLOR);
        }
    } else {
        camera = new pf::Camera();
    }

    // Event loop
    char method = 's';

    while (true) {

        if (!args.fromFile) {
            camera->capture(img);
        }

        // Rectify captures
        pf::StereoCapture stereo(img, params.map);
        stereo.displayRectified();

        // Compute disparity map
        pf::DisparityMap *dm;

        if (method == 'a') dm = new pf::BM(stereo);
        if (method == 's') dm = new pf::SGBM(stereo);
        if (method == 'd') dm = new pf::ELAS(stereo);

        dm->displayMap();
        // dm->displayHSV();

        delete dm;

        // Read keys
        char c = (char)cv::waitKey(30);
        if (c == 27) break;
        if (c == 'a' || c == 's' || c == 'd') method = c;
    }
}
