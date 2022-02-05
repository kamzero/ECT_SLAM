#include <opencv2/opencv.hpp>

#include "algorithm.hpp"
#include "backend.hpp"
#include "config.hpp"
#include "feature.hpp"
#include "frontend.hpp"
#include "map.hpp"
#include "viewer.hpp"
#include "orb.hpp"

namespace ECT_SLAM
{

    Frontend::Frontend()
    {
        orb_ = cv::ORB::create();
        num_features_init_ = Config::Get<int>("num_features_init");
        num_features_ = Config::Get<int>("num_features");
    }

    bool Frontend::AddFrame(ECT_SLAM::Frame::Ptr frame)
    {
        current_frame_ = frame;
        switch (status_)
        {
        case FrontendStatus::INITING:
            Init();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
        }

        last_frame_ = current_frame_;
        return true;
    }

    bool Frontend::Init()
    {
        std::cout << "Initing...\n";
        DetectFeature();

        status_ = FrontendStatus::TRACKING_GOOD;
        return true;
    }

    bool Frontend::Track()
    {
        std::cout << "Tracking No." << current_frame_->id_ << " ...\n";

        if (last_frame_)
        {
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
        }

        status_ = FrontendStatus::TRACKING_GOOD;

        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

        return true;
    }

    bool Frontend::Reset()
    {
        std::cout << "Reset is not implemented. " << std::endl;
        return true;
    }

    bool Frontend::DetectFeature(){
        std::vector<cv::KeyPoint> keypoints;
        std::vector<DescType> descriptors;
        cv::FAST(current_frame_->img_, keypoints, 40);
        ComputeORB(current_frame_->img_, keypoints, descriptors);

        for(int i = 0; i < keypoints.size(); i++){
            if(!descriptors[i].size()) continue;
            Feature::Ptr feature(new Feature(current_frame_, keypoints[i], descriptors[i]));
            current_frame_->features_.push_back(feature);
        }

        // std::cout << current_frame_->features_.size() << " / " <<  keypoints.size()<< std::endl;
        return true;
    }

} // namespace ECT_SLAM