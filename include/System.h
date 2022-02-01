#pragma once

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

namespace ECT_SLAM
{


class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        DVS=1
    };

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const std::string &strSettingsFile, const eSensor sensor);

    // Proccess the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);


    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectory(const std::string &filename);



private:


};

}// namespace ECT_SLAM

