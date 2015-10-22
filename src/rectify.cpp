#include <cstring>
#include <cassert>
#include "stereo.h"


#define WINDOW_SIZE 7


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


void mouseEvent(int event, int x, int y, int flags, void *data) {

    pf::DisparityMap *dm = (pf::DisparityMap*)data;

    if (dm != NULL)
    if (event == cv::EVENT_LBUTTONDOWN) {

        cv::Size size = dm->disparity.size();
        float max_element = 0;
        float min_element = INT_MAX;
        float all_elements = 0;
        int num_elements = 0;

        for (int dy = 0; dy < WINDOW_SIZE; dy++) {
            for (int dx = 0; dx < WINDOW_SIZE; dx++) {

                int fx = x + dx - (WINDOW_SIZE + 1) / 2;
                int fy = y + dy - (WINDOW_SIZE + 1) / 2;

                if (fy >= 0 && fy < size.height)
                if (fx >= 0 && fx < size.width) {

                    float value = dm->disparity.at<float>(fy, fx);

                    if (value) {
                        all_elements += value;
                        num_elements += 1;
                        if (value > max_element) max_element = value;
                        if (value < min_element) min_element = value;
                    }
                }
            }
        }

        std::cout << "Left button clicked at (" << x << ", " << y << ")" << std::endl;
        std::cout << "\tmin_element = " << min_element << std::endl;
        std::cout << "\tmax_element = " << max_element << std::endl;
        std::cout << "\tmean = " << (all_elements / (double) num_elements) << std::endl;
    }
}


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
        camera = new pf::VideoCamera();
    }

    // Event loop
    char method = 's';

    while (true) {

        std::cout << "--------" << std::endl;

        if (!args.fromFile) {
            camera->capture(img);
        }

        // Rectify captures
        pf::StereoCapture stereo(img, params.map);
        stereo.displayRectified();

        // Compute disparity map
        pf::DisparityMap *dm;

        clock_t begin = clock();

        if (method == 'a') dm = new pf::BM(stereo);
        if (method == 's') dm = new pf::SGBM(stereo);
        if (method == 'd') dm = new pf::ELAS(stereo);

        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

        std::cout << std::endl << "Time elapsed: " << elapsed_secs << std::endl;

        cv::setMouseCallback("Disparity Map", mouseEvent, dm);

        dm->displayMap();

        cv::Size size = dm->disparity.size();

        float min_val = 0;
        int valid = 0;

        for (int y = 0; y < size.height; y++) {
            for (int x = 0; x < size.width; x++) {
                if (dm->disparity.at<float>(y, x) < min_val)
                {
                    min_val = dm->disparity.at<float>(y, x);
                }
            }
        }

        for (int y = 0; y < size.height; y++) {
            for (int x = 0; x < size.width; x++) {
                if (dm->disparity.at<float>(y, x) != min_val)
                {
                    valid++;
                }
            }
        }

        double density = double(valid) / (size.height * size.width);

        std::cout << "Valid pixels: " << valid << std::endl;
        std::cout << "Density: " << density << std::endl;

        std::cout << "--------" << std::endl;

        // Read keys
        char c = (char)cv::waitKey(args.fromFile ? 0 : 30);

        delete dm;

        if (c == 27) break;
        if (c == 'a' || c == 's' || c == 'd') method = c;
    }
}
