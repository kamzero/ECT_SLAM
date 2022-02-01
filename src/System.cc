
#include "System.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

namespace ECT_SLAM
{

System::System(const std::string &strSettingsFile, const eSensor sensor)
{

    std::cout << "Input sensor was set to: ";

    if(sensor==MONOCULAR)
        std::cout << "Monocular" << std::endl;
    else if(sensor==DVS)
        std::cout << "DVS" << std::endl;

}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{

}

void System::Shutdown()
{

}

void System::SaveKeyFrameTrajectory(const std::string &filename)
{
    std::cout << std::endl << "Saving keyframe trajectory to " << filename << " ..." << std::endl;
}



} //namespace ECT_SLAM
