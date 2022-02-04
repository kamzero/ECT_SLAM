#include <opencv2/opencv.hpp>

#include "algorithm.hpp"
#include "backend.hpp"
#include "config.hpp"
#include "feature.hpp"
#include "frontend.hpp"
#include "map.hpp"
#include "viewer.hpp"

namespace ECT_SLAM {

Frontend::Frontend() {
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(ECT_SLAM::Frame::Ptr frame) {
    current_frame_ = frame;
    switch (status_) {
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

bool Frontend::Init() {
    std::cout << "Initing...\n";
    status_ = FrontendStatus::TRACKING_GOOD;
    return true;
}

bool Frontend::Track() {
    std::cout << "Tracking...\n";

    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    status_ = FrontendStatus::TRACKING_GOOD;

    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    return true;
}


bool Frontend::Reset() {
    std::cout << "Reset is not implemented. " << std::endl;
    return true;
}

}  // namespace ECT_SLAM