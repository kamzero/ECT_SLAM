#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <nmmintrin.h>

using namespace std;

// 32 bit unsigned int, will have 8, 8x32=256
typedef vector<uint32_t> DescType; // Descriptor type

/**
 * compute descriptor of orb keypoints
 * @param img input image
 * @param keypoints detected fast keypoints
 * @param descriptors descriptors
 *
 * NOTE: if a keypoint goes outside the image boundary (8 pixels), descriptors will not be computed and will be left as
 * empty
 */
void ComputeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors);

/**
 * brute-force match two sets of descriptors
 * @param desc1 the first descriptor
 * @param desc2 the second descriptor
 * @param matches matches of two images
 */
void BfMatch(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<cv::DMatch> &matches);
