#include "frame.hpp"

namespace ECT_SLAM {

Frame::Frame(long id, double time_stamp, const SE3 &pose, const Mat &img)
        : id_(id), time_stamp_(time_stamp), pose_(pose), img_(img) {}

Frame::Ptr Frame::CreateFrame() {
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id_ = factory_id++;
    return new_frame;
}

void Frame::SetKeyFrame() {
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}

}
