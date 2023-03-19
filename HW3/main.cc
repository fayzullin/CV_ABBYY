#include <cstdio>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "fht.h"


int main(int argc, char** argv) {
    if (argc != 3) {
        printf("use: %s <input_image_path> <output_radon_table_file>\n", argv[0]);
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];

    cv::Mat src = cv::imread(input);
    cv::Mat srcGray(src);
    cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);

    FastHoughTransformer fht(srcGray);
    const int slopeAngle = fht.GetDrtMaxDistortionAngle();
    fht.WriteDrt(output);

    return 0;
}
