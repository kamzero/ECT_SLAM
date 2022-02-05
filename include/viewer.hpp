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

   private:
    void ThreadLoop();

    Map::Ptr map_ = nullptr;

    std::thread viewer_thread_;
    bool viewer_running_ = true;

};
}  // namespace ECT_SLAM

