#pragma once

#include <opencv2/opencv.hpp>

class FastHoughTransformer {
public:
    explicit FastHoughTransformer(const cv::Mat &I);

    int GetDrtMaxDistortionAngle() const;

    void WriteDrt(const char *fileName) const;

private:
    const int n;
    const int ticksInDegree;
    cv::Mat drt; // discrete Radon transformation table

    void FastHoughTransform(const cv::Mat &I);

    cv::Mat FastHoughTransformFrom0ToPiDiv4(const cv::Mat &I);
};
