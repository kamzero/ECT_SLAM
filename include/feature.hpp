#pragma once

#include <memory>
#include <opencv2/features2d.hpp>
#include "common_include.hpp"

namespace ECT_SLAM
{

    struct Frame;
    struct MapPoint;
    
    enum STATUS{
        NONMATCH=0,
        MATCH3D=1,
        MATCH2D=2
    };

    /**
 * 2D 特征点
 * 在三角化之后会被关联一个地图点
 */
    struct Feature
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Feature> Ptr;

        std::weak_ptr<Frame> frame_;        // 持有该feature的frame
        cv::KeyPoint position_;             // 2D提取位置
        std::weak_ptr<MapPoint> map_point_; // 关联地图点

        bool is_outlier_ = false; // 是否为异常点
        STATUS status_ = STATUS::NONMATCH;

        //! TODO: timestamp

    public:
        Feature() {}

        Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
            : frame_(frame), position_(kp) {}

    };
} // namespace ECT_SLAM
