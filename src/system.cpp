
#include "system.hpp"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

namespace ECT_SLAM
{

    System::System(const std::string &config_path, const eSensor sensor) : config_file_path_(config_path), sensor_(sensor)
    {
    }

    bool System::Init()
    {

        // read from config file
        if (Config::SetParameterFile(config_file_path_) == false)
        {
            return false;
        }

        std::cout << "The input sensor was set to: ";

        if (sensor_ == MONOCULAR)
            std::cout << "Monocular" << std::endl;
        else if (sensor_ == DVS)
            std::cout << "DVS" << std::endl;

        // create components and links
        frontend_ = Frontend::Ptr(new Frontend);
        backend_ = Backend::Ptr(new Backend);
        map_ = Map::Ptr(new Map);
        viewer_ = Viewer::Ptr(new Viewer);
        camera_ = Camera::Ptr(new Camera(Config::Get<double>("camera.fx"), Config::Get<double>("camera.fy"),
                                      Config::Get<double>("camera.cx"), Config::Get<double>("camera.cy")));

        frontend_->SetBackend(backend_);
        frontend_->SetMap(map_);
        frontend_->SetViewer(viewer_);
        frontend_->SetCameras(camera_);

        backend_->SetMap(map_);
        backend_->SetCameras(camera_);

        viewer_->SetMap(map_);

        return true;
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
    {
        return im;
    }

    void System::Shutdown()
    {
    }

    void System::SaveKeyFrameTrajectory(const std::string &filename)
    {
        std::cout << std::endl
                  << "Saving keyframe trajectory to " << filename << " ..." << std::endl;
    }

} //namespace ECT_SLAM
