
#include "viewer.hpp"
#include "feature.hpp"
#include "frame.hpp"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace ECT_SLAM {

Viewer::Viewer() {
    viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
}
void Viewer::ThreadLoop() {
    while (!viewer_running_) {
        usleep(5000);
    }

    std::cout << "Stop viewer\n";
}

}  // namespace ECT_SLAM
