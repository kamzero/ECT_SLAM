#pragma once

#include "camera.hpp"
#include "common_include.hpp"

namespace ECT_SLAM {

// forward declare
struct MapPoint;
struct Feature;

/**
 * 帧
 * 每一帧分配独立id，关键帧分配关键帧ID
 */
struct Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;           // id of this frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 是否为关键帧
    double time_stamp_;              // 时间戳，暂不使用
    SE3 pose_;                       // Tcw 形式Pose
    std::mutex pose_mutex_;          // Pose数据锁
    cv::Mat img_;   // images

    // extracted features in image
    std::vector<std::shared_ptr<Feature>> features_;
    // std::vector<DescType> descriptors_;
    cv::Mat descriptors_;


   public:  // data members
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, const Mat &img);

    // set and get pose, thread safe
    SE3 Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    Mat34f RT(){
        return Pose().matrix().block<3, 4>(0, 0).cast <float>();
    }

    void SetPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    /// 设置关键帧并分配并键帧id
    void SetKeyFrame();

    /// 工厂构建模式，分配id 
    static std::shared_ptr<Frame> CreateFrame();
};

}  // namespace ECT_SLAM
