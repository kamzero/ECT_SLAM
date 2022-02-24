#pragma once

#include <opencv2/features2d.hpp>

#include "common_include.hpp"
#include "frame.hpp"
#include "map.hpp"

namespace ECT_SLAM
{

   class Backend;
   class Viewer;

   enum class FrontendStatus
   {
      INITING,
      TRACKING_GOOD,
      TRACKING_BAD,
      LOST
   };

   /**
 * 前端
 * 估计当前帧Pose，在满足关键帧条件时向地图加入关键帧并触发优化
 */
   class Frontend
   {
   public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      typedef std::shared_ptr<Frontend> Ptr;

      Frontend();

      /// 外部接口，添加一个帧并计算其定位结果
      bool AddFrame(Frame::Ptr frame);

      /// Set函数
      void SetMap(Map::Ptr map) { map_ = map; }

      void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

      void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }

      FrontendStatus GetStatus() const { return status_; }

      void SetCameras(Camera::Ptr cam)
      {
         camera_ = cam;
      }

   private:
      /**
     * Initialize
     * @return true if success
     */
      bool Init();

      /**
     * Track in normal mode
     * @return true if success
     */
      bool Track();

      /**
     * Reset when lost
     * @return true if success
     */
      bool Reset();

      /**
     * Detect Feature & Calculate ORB Descriptors
     * @return true if success
     */
      bool DetectFeature();

      bool MatchWith3DMap(std::vector<cv::DMatch> &matches, std::vector<cv::Point3d> &points_3d, std::vector<cv::Point2d> &points_2d);

      bool MatchAndBuildMap(Frame::Ptr frame1, Frame::Ptr frame2);

      bool MatchAndUpdateMap(Frame::Ptr frame1, Frame::Ptr frame2);

      bool Match2D2D(Frame::Ptr &frame1, Frame::Ptr frame2, std::vector<cv::DMatch> &matches,
                     std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

      bool Trangulation(Frame::Ptr &frame1, Frame::Ptr frame2,
                        std::vector<cv::DMatch> &matches,
                        std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

      bool EstimateWithMatches(Frame::Ptr &frame1, Frame::Ptr frame2,
                               std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

      // data
      FrontendStatus status_ = FrontendStatus::INITING;

      Frame::Ptr current_frame_ = nullptr; // 当前帧
      Frame::Ptr last_frame_ = nullptr;    // 上一帧
      Frame::Ptr first_frame_ = nullptr;   // 第0帧

      Camera::Ptr camera_ = nullptr; // 左侧相机

      Map::Ptr map_ = nullptr;
      std::shared_ptr<Backend> backend_ = nullptr;
      std::shared_ptr<Viewer> viewer_ = nullptr;

      SE3 relative_motion_; // 当前帧与上一帧的相对运动，用于估计当前帧pose初值

      int tracking_inliers_ = 0; // inliers, used for testing new keyframes

      // params
      int num_features_ = 200;
      int num_features_init_ = 100;
      int num_features_tracking_ = 50;
      int num_features_tracking_bad_ = 20;
      double ratio_for_keyframe_ = 0.40;

      // utilities
      cv::Ptr<cv::ORB> orb_; // feature detector in opencv
   };

   void BfMatch3D(const ECT_SLAM::Map::LandmarksType &landmarks, const vector<DescType> &desc, vector<cv::DMatch> &matches);

} // namespace ECT_SLAM
