
#include "system.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

namespace ECT_SLAM
{

System::System(const std::string &config_path, const eSensor sensor):
config_file_path_(config_path), sensor_(sensor)
{}

bool System::Init() {

    // read from config file
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }

    std::cout << "The input sensor was set to: ";

    if(sensor_==MONOCULAR)
        std::cout << "Monocular" << std::endl;
    else if(sensor_==DVS)
        std::cout << "DVS" << std::endl;

    
    return true;
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
