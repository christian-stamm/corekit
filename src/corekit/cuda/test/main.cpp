#include <opencv2/core/hal/interface.h>

#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "corekit/cuda/image.hpp"
using namespace corekit::cuda;

int main() {
    uint8_t* yuv = nullptr;

    cv::Mat img =
        cv::imread("/home/orinagx/Documents/trinity-visu/res/media/test.jpg");

    cv::resize(img, img, cv::Size(1920, 1080));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    Image3U input = Image3U::fromCvMat(img);

    yuv = input.toNv16(yuv, false);

    Image3U converted = Image3U::fromNv16(yuv, make_uint2(1920, 1080), false);

    cv::Mat output = converted.toCvMat();

    cv::imshow("Output", output);
    cv::waitKey(10000);

    return 0;
}