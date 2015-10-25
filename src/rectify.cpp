#include <cstring>
#include <cassert>
#include "stereo.h"


#define WINDOW_SIZE 7

#define DENSITY_LEFT 10
#define DENSITY_RIGHT 10
#define DENSITY_TOP 10
#define DENSITY_BOTTOM 10


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


void computeMouseWindow(int x, int y, pf::DisparityMap& map) {

    cv::Size size = map.disparity.size();
    float max_element = FLT_MIN;
    float min_element = FLT_MAX;
    float all_elements = 0;
    int num_elements = 0;

    for (int dy = 0; dy < WINDOW_SIZE; dy++) {
        for (int dx = 0; dx < WINDOW_SIZE; dx++) {

            int fx = x + dx - (WINDOW_SIZE + 1) / 2;
            int fy = y + dy - (WINDOW_SIZE + 1) / 2;

            if (fy >= 0 && fy < size.height)
            if (fx >= 0 && fx < size.width) {

                float value = map.disparity.at<float>(fy, fx);

                if (value) {
                    all_elements += value;
                    num_elements += 1;
                    if (value > max_element) max_element = value;
                    if (value < min_element) min_element = value;
                }
            }
        }
    }

    //std::cout << "Left button clicked at (" << x << ", " << y << ")" << std::endl;
    std::cout << "\t" << x << "\t" << y;
    std::cout << "\t" << min_element;
    std::cout << "\t" << max_element;
    std::cout << "\t" << (all_elements / (double) num_elements) << std::endl;
}


void computeDensity(pf::DisparityMap& map) {

    cv::Size size = map.disparity.size();

    float min_val = 0;
    int valid = 0;
    int all = 0;

    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            if (map.disparity.at<float>(y, x) < min_val) {
                min_val = map.disparity.at<float>(y, x);
            }
        }
    }

    for (int y = DENSITY_TOP; y < size.height - DENSITY_BOTTOM; y++) {
        for (int x = DENSITY_LEFT; x < size.width - DENSITY_RIGHT; x++) {
            all++;
            if (map.disparity.at<float>(y, x) != min_val) {
                valid++;
            }
        }
    }

    double density = double(valid) / all;
    std::cout << "\t" << density;
}


void mouseEvent(int event, int x, int y, int flags, void *data) {

    pf::StereoCapture *stereo = (pf::StereoCapture*)data;

    pf::DisparityMap *algs[] = {
        new pf::BM(*stereo),
        new pf::SGBM(*stereo),
        new pf::ELAS(*stereo),
    };

    if (stereo != NULL && event == cv::EVENT_LBUTTONDOWN) {

        for (auto map: algs) {

            //std::cout << "Method: " << map->name << std::endl;

            clock_t begin = clock();
            map->compute();
            clock_t end = clock();

            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            //std::cout << "Time: " << elapsed_secs << std::endl;
            std::cout << elapsed_secs;

            computeDensity(*map);
            computeMouseWindow(x, y, *map);

            delete map;
        }
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
    pf::DisparityMap *dm;
    pf::StereoCapture::Plane p;

	char plane = 'q';
    char method = 's';

    while (true) {

        if (!args.fromFile) {
            camera->capture(img);
        }

		// Select color plane
        if (plane == 'q') p = pf::StereoCapture::ALL;
        if (plane == 'w') p = pf::StereoCapture::RED;
        if (plane == 'e') p = pf::StereoCapture::GREEN;
        if (plane == 'r') p = pf::StereoCapture::BLUE;

        // Rectify captures
        pf::StereoCapture stereo(img, params.map, p);
        stereo.displayRectified();

        // Set mouse event listener
        cv::setMouseCallback("Disparity Map", mouseEvent, &stereo);

        // Compute disparity map
        if (method == 'a') dm = new pf::BM(stereo);
        if (method == 's') dm = new pf::SGBM(stereo);
        if (method == 'd') dm = new pf::ELAS(stereo);

        dm->compute();
        dm->displayMap();

        // Read keys
        char c = (char)cv::waitKey(args.fromFile ? 0 : 30);

        delete dm;

        if (c == 27) break;
        if (c == 'a' || c == 's' || c == 'd') method = c;
		if (c == 'q' || c == 'w' || c == 'e' || c == 'r') plane = c;
    }
}
