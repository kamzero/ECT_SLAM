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
        if (current_frame_->id_ == 0)
            first_frame_ = frame;

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

        if (current_frame_->id_ >= 5)
        {
            if (MatchAndBuildMap(first_frame_, current_frame_))
                status_ = FrontendStatus::TRACKING_GOOD;
        }

        return true;
    }

    bool Frontend::Track()
    {
        std::cout << "Tracking No." << current_frame_->id_ << " ...\n";

        DetectFeature();
        // initial guess
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());


        //!--------------Add New MapPoints With 2D-2D Matches(last frame)--------------
        MatchAndBuildMap(last_frame_, current_frame_);

        // end stage
        status_ = FrontendStatus::TRACKING_GOOD;
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

        return true;
    }

    bool Frontend::Reset()
    {
        std::cout << "Reset is not implemented. " << std::endl;
        return true;
    }

    bool Frontend::DetectFeature()
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(current_frame_->img_, keypoints, 40);
        ComputeORB(current_frame_->img_, keypoints, current_frame_->descriptors_);

        for (int i = 0; i < keypoints.size(); i++)
        {
            Feature::Ptr feature(new Feature(current_frame_, keypoints[i]));
            current_frame_->features_.push_back(feature);
        }

        return true;
    }

    bool Frontend::Match2D2D(Frame::Ptr &frame1, Frame::Ptr frame2,
                             std::vector<cv::DMatch> &matches,
                             std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2)
    {
        BfMatch(frame1->descriptors_, frame2->descriptors_, matches);
        if (matches.size() < 8)
            return false;
        for (auto m : matches)
        {
            points1.emplace_back(frame1->features_[m.queryIdx]->position_.pt.x, frame1->features_[m.queryIdx]->position_.pt.y);
            points2.emplace_back(frame2->features_[m.trainIdx]->position_.pt.x, frame2->features_[m.trainIdx]->position_.pt.y);
        }
        return true;
    }

    bool Frontend::Trangulation(Frame::Ptr &frame1, Frame::Ptr frame2,
                                std::vector<cv::DMatch> &matches,
                                std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2)
    {
        cv::Mat P1, P2;
        Mat34f P1_e = camera_->K() * frame1->RT();
        Mat34f P2_e = camera_->K() * frame2->RT();
        cv::eigen2cv(P1_e, P1);
        cv::eigen2cv(P2_e, P2);

        // std::cout << "-----first-----\n"<< P1<< std::endl;
        // std::cout << "----current----\n"<< P2 << std::endl;

        cv::Mat pointsH(1, matches.size(), CV_32FC4);
        cv::Mat points3F;
        cv::triangulatePoints(P1, P2, points1, points2, pointsH);
        cv::Mat pointsCH = pointsH.reshape(4);
        std::cout << "----map_points CH----\n" << pointsCH.rows << " " << pointsCH.cols << " " << pointsCH.channels() << " " << pointsCH.type() << std::endl;
        cv::convertPointsFromHomogeneous(pointsCH, points3F);
        std::cout << "----map_points 3F----\n" << points3F.rows << " " << points3F.cols << " " << points3F.channels() << " " << points3F.type() << std::endl;

        SE3 pose_Tcw = frame1->Pose().inverse();
        for (int i = 0; i < matches.size(); i++)
        {
            if (points3F.at<cv::Vec3f>(i, 0)[2] <= 0)
                continue;

            Vec3 pworld = Vec3(points3F.at<cv::Vec3f>(i, 0)[0], points3F.at<cv::Vec3f>(i, 0)[1], points3F.at<cv::Vec3f>(i, 0)[2]);

            auto new_map_point = MapPoint::CreateNewMappoint();
            pworld = pose_Tcw * pworld;
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(frame1->features_[matches[i].queryIdx]);
            new_map_point->AddObservation(
                frame2->features_[matches[i].trainIdx]);

            frame1->features_[matches[i].queryIdx]->map_point_ = new_map_point;
            frame2->features_[matches[i].trainIdx]->map_point_ = new_map_point;
            map_->InsertMapPoint(new_map_point);
        }
        return true;
    }

    bool Frontend::EstimateWithMatches(Frame::Ptr &frame1, Frame::Ptr frame2,
                                       std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2)
    {
        cv::Mat K = camera_->K_cv();
        cv::Mat E, R, t, r;
        E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 1.0);
        cv::recoverPose(E, points1, points2, K, R, t);
        // cv::Rodrigues(r, R);

        Mat33 R_e;
        Vec3 t_e;
        cv::cv2eigen(R, R_e);
        cv::cv2eigen(t, t_e);
        SE3 pose(R_e, t_e);

        frame2->SetPose(pose * frame1->Pose());

        // std::cout << "------R------\n"<<R << std::endl;
        // std::cout << "-----R_e-----\n"<<R_e << std::endl;
        // std::cout << "------t------\n"<<t <<std::endl;
        // std::cout << "-----t_e------\n"<<t_e <<std::endl;
        // std::cout << "-----se3-----\n"<< pose.log() << std::endl;
        std::cout << "-----SE3-----\n"
                  << pose.matrix() << std::endl;
        return true;
    }

    bool Frontend::MatchAndBuildMap(Frame::Ptr frame1, Frame::Ptr frame2)
    {
        std::vector<cv::DMatch> matches;
        std::vector<cv::Point2f> points1, points2;
        //!-----------------------Match----------------------------
        if (!Match2D2D(frame1, frame2, matches, points1, points2))
            return false;

        //!-----------------------Estimate----------------------------
        EstimateWithMatches(frame1, frame2, points1, points2);

        //!-----------------------Trangulation & Build Map From 2D-2D Matches----------------------------
        Trangulation(frame1, frame2, matches, points1, points2);

        return true;
    }

} // namespace ECT_SLAM