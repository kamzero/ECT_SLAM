
#include "viewer.hpp"
#include "feature.hpp"
#include "frame.hpp"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace ECT_SLAM
{

    Viewer::Viewer()
    {
        viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
    }

    void Viewer::AddCurrentFrame(Frame::Ptr current_frame)
    {
        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        current_frame_ = current_frame;
    }
    
    void Viewer::ThreadLoop()
    {
        pangolin::CreateWindowAndBind("ECT_SLAM", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState vis_camera(
            pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View &vis_display =
            pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                .SetHandler(new pangolin::Handler3D(vis_camera));

        const float blue[3] = {0, 0, 1};
        const float green[3] = {0, 1, 0};

        while (!pangolin::ShouldQuit() && viewer_running_)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            vis_display.Activate(vis_camera);

            std::unique_lock<std::mutex> lock(viewer_data_mutex_);
            if (current_frame_)
            {
                DrawFrame(current_frame_, green);
                FollowCurrentFrame(vis_camera);

                cv::Mat img = PlotFrameImage();
                cv::imshow("image", img);
                cv::waitKey(1);
            }

            if (map_)
            {
                DrawMapPoints();
            }

            pangolin::FinishFrame();
            usleep(5000);
        }
        std::cout << "Stop viewer\n";
    }

    void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState &vis_camera)
    {
        SE3 Twc = current_frame_->Pose().inverse();
        pangolin::OpenGlMatrix m(Twc.matrix());
        vis_camera.Follow(m, true);
    }

    cv::Mat Viewer::PlotFrameImage()
    {
        cv::Mat img_out;
        auto cf = current_frame_;
        cv::cvtColor(cf->img_, img_out, CV_GRAY2BGR);
        for (size_t i = 0; i < cf->features_.size(); ++i)
        {
            if (cf->features_[i]->map_point_.lock())
            {
                auto feat = cf->features_[i];
                cv::circle(img_out, feat->position_.pt, 1, cv::Scalar(0, 250, 0),2);
            }
            else{
                auto feat = cf->features_[i];
                cv::circle(img_out, feat->position_.pt, 1, cv::Scalar(0, 0, 250),2);
            }
        }
        return img_out;
    }

    void Viewer::DrawFrame(Frame::Ptr frame, const float *color)
    {
        SE3 Twc = frame->Pose().inverse();
        const float sz = 1.0;
        const int line_width = 2.0;
        const float fx = 400;
        const float fy = 400;
        const float cx = 512;
        const float cy = 384;
        const float width = 1080;
        const float height = 768;

        glPushMatrix();

        Sophus::Matrix4f m = Twc.matrix().template cast<float>();
        glMultMatrixf((GLfloat *)m.data());

        if (color == nullptr)
        {
            glColor3f(1, 0, 0);
        }
        else
            glColor3f(color[0], color[1], color[2]);

        glLineWidth(line_width);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glEnd();
        glPopMatrix();
    }

    void Viewer::DrawMapPoints()
    {
        const float red[3] = {1.0, 0, 0};
        for (auto &kf : active_keyframes_)
        {
            DrawFrame(kf.second, red);
        }

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &landmark : active_landmarks_)
        {
            auto pos = landmark.second->Pos();
            glColor3f(red[0], red[1], red[2]);
            glVertex3d(pos[0], pos[1], pos[2]);
        }
        glEnd();
    }

    void Viewer::Close()
    {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void Viewer::UpdateMap()
    {
        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        assert(map_ != nullptr);
        active_keyframes_ = map_->GetActiveKeyFrames();
        active_landmarks_ = map_->GetActiveMapPoints();
        map_updated_ = true;
    }

} // namespace ECT_SLAM
