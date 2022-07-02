
#include "system.hpp"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
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

        if (viewer_)
            viewer_->SetMap(map_);

        return true;
    }

    void System::ProcessImage(std::string & image_path, const double & timestamp){
        cv::Mat image = cv::imread(image_path, 0);
        cv::Mat dst;
        cv::resize(image, dst, cv::Size(640, 480));
        TrackMonocular(dst, timestamp);
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
    {
        auto new_frame = Frame::CreateFrame();
        // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        cv::equalizeHist(im, new_frame->img_);
        // clahe->apply(src, new_frame->img_);
        new_frame->time_stamp_ = timestamp;
        bool success = frontend_->AddFrame(new_frame);
        return im;
    }

    void System::Shutdown()
    {
        backend_->Stop();
        if (viewer_)
            viewer_->Close();
    }

    void System::SaveKeyFrameTrajectory(const std::string &filename)
    {
        std::cout << "\nSaving keyframe trajectory to " << filename << " ..." << std::endl;

        Map::KeyframesType keyframes = map_->GetAllKeyFrames();

        std::ofstream f;
        f.open(filename.c_str());
        f << std::fixed;

        for (auto kf : keyframes)
        {
            double time = kf.second->time_stamp_;
            SE3 pose = kf.second->Pose();
            auto se3 = pose.log();
            std::vector<double> rvec{se3(3, 0), se3(4, 0), se3(5, 0)};
            std::vector<double> tvec{se3(0, 0), se3(1, 0), se3(2, 0)};

            Eigen::Quaterniond qt;
            qt = Eigen::AngleAxisd(rvec[0], Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(rvec[1], Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(rvec[2], Eigen::Vector3d::UnitX());

            f << time << " " << tvec[0] << " " << tvec[1] << " " << tvec[2] << " "
              << qt.x() << " " << qt.y() << " " << qt.z() << " " << qt.w() << "\n";
        }

        f.close();
        std::cout << "trajectory file close..." << std::endl;
    }

} //namespace ECT_SLAM
