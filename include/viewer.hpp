//
// Created by gaoxiang on 19-5-4.
//

#include <thread>
#include <pangolin/pangolin.h>

#include "common_include.hpp"
#include "frame.hpp"
#include "map.hpp"

namespace ECT_SLAM {

/**
 * 可视化
 */
class Viewer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer();

    void SetMap(Map::Ptr map) { map_ = map; }

    void Close();

    // 增加一个当前帧
    void AddCurrentFrame(Frame::Ptr current_frame);

    // 更新地图
    void UpdateMap();

   private:
    void ThreadLoop();

    void DrawFrame(Frame::Ptr frame, const float* color);

    void DrawMapPoints();

    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    /// plot the features in current frame into an image
    cv::Mat PlotFrameImage();

    Frame::Ptr current_frame_ = nullptr;
    Map::Ptr map_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true;
    
    Map::KeyframesType active_keyframes_;
    Map::LandmarksType active_landmarks_;
    bool map_updated_ = false;

    std::mutex viewer_data_mutex_;
};
}  // namespace ECT_SLAM

