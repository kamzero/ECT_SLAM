#include <opencv2/opencv.hpp>

#include "algorithm.hpp"
#include "backend.hpp"
#include "config.hpp"
#include "feature.hpp"
#include "frontend.hpp"
#include "map.hpp"
#include "viewer.hpp"

namespace ECT_SLAM
{

    Frontend::Frontend()
    {
        gftt_ =
            cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);

        detector_ = cv::ORB::create();
        descriptor_ = cv::ORB::create();
        matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");

        num_features_init_ = Config::Get<int>("num_features_init");
        num_features_ = Config::Get<int>("num_features");
        num_for_keyframe_ = Config::Get<int>("num_for_keyframe");
        ratio_for_keyframe_ = Config::Get<double>("ratio_for_keyframe");
    }

    bool Frontend::AddFrame(ECT_SLAM::Frame::Ptr frame)
    {
        last_frame_ = current_frame_;
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

        return true;
    }

    bool Frontend::Init()
    {
        std::cout << "Initing...\n";

        DetectFeature();

        if (current_frame_->id_ >= 5)
        {
            if (MatchAndBuildMap(first_frame_, current_frame_))
            {
                status_ = FrontendStatus::TRACKING_GOOD;
                if (viewer_)
                {
                    viewer_->AddCurrentFrame(current_frame_);
                    viewer_->UpdateMap();
                }
            }
        }

        return true;
    }

    bool Frontend::MatchLastFrame(std::vector<cv::Point3d> &points_3d, std::vector<cv::Point2d> &points_2d,
                                  std::vector<cv::Point2f> &last_to_be_tri, std::vector<cv::Point2f> &current_to_be_tri,
                                  std::vector<cv::DMatch> &matches)
    {
        std::vector<cv::DMatch> bf_matches;
        matcher_->match(last_frame_->descriptors_, current_frame_->descriptors_, bf_matches);

        int num_good_pts = 0;
        for (auto m : bf_matches)
        {
            auto last_feature = last_frame_->features_[m.queryIdx];
            auto current_feature = current_frame_->features_[m.trainIdx];

            if (abs(last_feature->position_.pt.x - current_feature->position_.pt.x) > 40 || abs(last_feature->position_.pt.y - current_feature->position_.pt.y) > 40)
                continue;
            current_feature->map_point_ = last_feature->map_point_;
            if (auto mp = last_feature->map_point_.lock())
            {
                num_good_pts++;
                auto pt3d = mp->Pos();
                points_3d.emplace_back(pt3d[0], pt3d[1], pt3d[2]);
                points_2d.emplace_back(current_feature->position_.pt.x, current_feature->position_.pt.y);

                // Add Obs
                mp->AddObservation(current_feature);
                current_feature->status_ = STATUS::MATCH3D;
            }
            else
            {
                last_to_be_tri.emplace_back(last_feature->position_.pt.x, last_feature->position_.pt.y);
                current_to_be_tri.emplace_back(current_feature->position_.pt.x, current_feature->position_.pt.y);
                matches.push_back(m);

                current_feature->status_ = STATUS::MATCH2D;
            }
        }
        double ratio = (double)num_good_pts / (double)(bf_matches.size());
        LOG(INFO) << "ratio: " << num_good_pts << "/" << bf_matches.size() << " = " << ratio << ". ";

        if (ratio < ratio_for_keyframe_)
            return true;
        if (num_good_pts < num_for_keyframe_)
            return true;
        return false;
    }

    bool Frontend::TrackLastFrame(std::vector<cv::Point3d> &points_3d, std::vector<cv::Point2d> &points_2d,
                                  std::vector<cv::Point2f> &last_to_be_tri, std::vector<cv::Point2f> &current_to_be_tri,
                                  std::vector<cv::DMatch> &matches)

    {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &kp : last_frame_->features_)
        {
            // if (kp->map_point_.lock())
            // {
            //     // use project point
            //     auto mp = kp->map_point_.lock();
            //     auto px =
            //         camera_->world2pixel(mp->pos_, current_frame_->Pose());
            //     kps_last.push_back(kp->position_.pt);
            //     kps_current.push_back(cv::Point2f(px[0], px[1]));
            // }
            // else
            {
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(
            last_frame_->img_, current_frame_->img_, kps_last,
            kps_current, status, error, cv::Size(11, 11), 1,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                             0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;
        int k = 0; // count feature in current_frame_
        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i])
            {
                cv::KeyPoint kp(kps_current[i], 7);
                Feature::Ptr feature(new Feature(current_frame_, kp));
                feature->map_point_ = last_frame_->features_[i]->map_point_;
                current_frame_->features_.push_back(feature);

                if (auto mp = feature->map_point_.lock())
                {
                    num_good_pts++;
                    auto pt3d = mp->Pos();
                    points_3d.emplace_back(pt3d[0], pt3d[1], pt3d[2]);
                    points_2d.push_back(kps_current[i]);

                    // Add Obs
                    mp->AddObservation(feature);
                    feature->status_ = STATUS::MATCH3D;
                }
                else
                {
                    last_to_be_tri.push_back(kps_last[i]);
                    current_to_be_tri.push_back(kps_current[i]);
                    matches.push_back(cv::DMatch{i, k, 0});

                    feature->status_ = STATUS::MATCH2D;
                }
                k++;
            }
        }

        double ratio = (double)num_good_pts / (double)(kps_last.size());
        LOG(INFO) << "ratio: " << num_good_pts << "/" << kps_last.size() << " = " << ratio << ". ";

        if (ratio < ratio_for_keyframe_)
            return true;
        if (num_good_pts < num_for_keyframe_)
            return true;
        return false;
    }

    bool Frontend::EstimatePnP(std::vector<cv::Point3d> &points_3d, std::vector<cv::Point2d> &points_2d)
    {
        if (points_3d.size() < 5)
            return false;
        auto pose = current_frame_->Pose();
        auto se3 = pose.log();
        std::vector<double> rvec{se3(3, 0), se3(4, 0), se3(5, 0)};
        std::vector<double> tvec{se3(0, 0), se3(1, 0), se3(2, 0)};

        cv::solvePnP(points_3d, points_2d, camera_->K_cv(), cv::Mat(), rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
        Vec6 new_se3;
        new_se3 << tvec[0], tvec[1], tvec[2], rvec[0], rvec[1], rvec[2];
        current_frame_->SetPose(SE3::exp(new_se3));

        std::cout << "=POSE= " << rvec[0] << " " << rvec[1] << " " << rvec[2] << " "
                  << tvec[0] << " " << tvec[1] << " " << tvec[2] << std::endl;
        return true;
    }

    bool Frontend::Track()
    {
        std::cout << "Tracking No." << current_frame_->id_ << " ... ";

        // initial guess
        current_frame_->SetPose(last_frame_->Pose());
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());

        //!--------------PnP Estimate With 2D-3D Matches(map)--------------------------
        std::vector<cv::DMatch> matches;
        std::vector<cv::Point2d> points_2d;
        std::vector<cv::Point3d> points_3d;
        std::vector<cv::Point2f> last_to_be_tri;
        std::vector<cv::Point2f> current_to_be_tri;

        DetectFeature();

        bool is_keyframe;
        is_keyframe = MatchLastFrame(points_3d, points_2d, last_to_be_tri, current_to_be_tri, matches);

        EstimatePnP(points_3d, points_2d);

        // //!--------------Add New MapPoints With 2D-2D Matches(last frame)--------------
        if (is_keyframe)
        {
            InsertKeyFrame(last_to_be_tri, current_to_be_tri, matches);
        }

        // end stage
        status_ = FrontendStatus::TRACKING_GOOD;
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();
        if (viewer_)
            viewer_->AddCurrentFrame(current_frame_);
        return true;
    }

    bool Frontend::DetectAndTriNewFeature()
    {
        std::vector<cv::Point2f> points1, points2;

        cv::Mat mask(last_frame_->img_.size(), CV_8UC1, 255);
        for (auto &feat : last_frame_->features_)
        {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                          feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
        }
        std::vector<cv::KeyPoint> keypoints, final_kps;
        gftt_->detect(last_frame_->img_, keypoints, mask);

        std::vector<cv::Point2f> kps_last, kps_current;

        for (auto &kp : keypoints)
        {
            kps_last.push_back(kp.pt);
            kps_current.push_back(kp.pt);
            final_kps.push_back(kp);
        }

        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(
            last_frame_->img_, current_frame_->img_, kps_last,
            kps_current, status, error, cv::Size(11, 11), 1,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                             0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i])
            {
                points1.push_back(kps_last[i]);
                points2.push_back(kps_current[i]);
            }
        }

        if (!points2.size())
            return false;

        cv::Mat P1, P2;
        Mat34f P1_e = camera_->K() * last_frame_->RT();
        Mat34f P2_e = camera_->K() * current_frame_->RT();
        cv::eigen2cv(P1_e, P1);
        cv::eigen2cv(P2_e, P2);

        cv::Mat pointsH(1, points2.size(), CV_32FC4);
        cv::Mat points3F;

        cv::triangulatePoints(P1, P2, points1, points2, pointsH);
        cv::convertPointsFromHomogeneous(pointsH.t(), points3F);

        SE3 pose_Tcw = last_frame_->Pose().inverse();
        for (int i = 0; i < points2.size(); i++)
        {
            if (points3F.at<cv::Vec3f>(i, 0)[2] <= 0)
                continue;

            Vec3 pworld = Vec3(points3F.at<cv::Vec3f>(i, 0)[0], points3F.at<cv::Vec3f>(i, 0)[1], points3F.at<cv::Vec3f>(i, 0)[2]);

            Feature::Ptr current_feature(new Feature(current_frame_, final_kps[i]));
            current_frame_->features_.push_back(current_feature);

            // auto new_map_point = MapPoint::CreateNewMappoint();
            // pworld = pose_Tcw * pworld;
            // new_map_point->SetPos(pworld);
            // new_map_point->AddObservation(current_feature);

            // current_feature->map_point_ = new_map_point;
            current_feature->status_ = STATUS::MATCH2D;

            // map_->InsertMapPoint(new_map_point);
        }
        std::cout << "DetectAndAdd " << points2.size() << " new features\n";

        return true;
    }

    bool Frontend::InsertKeyFrame(std::vector<cv::Point2f> &last_to_be_tri, std::vector<cv::Point2f> &current_to_be_tri,
                                  std::vector<cv::DMatch> &matches)
    {
        //!---------------------Triangulate-------------------
        Trangulation(last_frame_, current_frame_, matches, last_to_be_tri, current_to_be_tri);

        //!-------------------End Stage------------------------
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);
        backend_->UpdateMap();
        if (viewer_)
        {
            viewer_->UpdateMap();
        }
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
        descriptor_->compute(current_frame_->img_, keypoints, current_frame_->descriptors_);

        for (int i = 0; i < keypoints.size(); i++)
        {
            Feature::Ptr feature(new Feature(current_frame_, keypoints[i]));
            current_frame_->features_.push_back(feature);
        }

        return true;
    }

    bool Frontend::Match2D2D(Frame::Ptr &frame1, Frame::Ptr frame2,
                             std::vector<cv::DMatch> &matches,
                             std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, int thres = 8)
    {
        matcher_->match(frame1->descriptors_, frame2->descriptors_, matches);
        if (matches.size() < thres)
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
        if (matches.size() == 0)
            return true;

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
        cv::convertPointsFromHomogeneous(pointsH.t(), points3F);
        // std::cout << "----map_points 3F----\n" << points3F.rows << " " << points3F.cols << " " << points3F.channels() << " " << points3F.type() << std::endl;

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
            new_map_point->AddObservation(frame2->features_[matches[i].trainIdx]);

            frame1->features_[matches[i].queryIdx]->map_point_ = new_map_point;
            frame2->features_[matches[i].trainIdx]->map_point_ = new_map_point;
            frame2->features_[matches[i].trainIdx]->status_ = STATUS::MATCH2D;

            map_->InsertMapPoint(new_map_point);
        }
        std::cout << "Trangulation " << matches.size() << " new mappoints\n";
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
        std::cout << "-----se3-----\n"
                  << pose.log() << std::endl;
        // std::cout << "-----SE3-----\n"
        //           << pose.matrix() << std::endl;
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

        //!-----------------------Insert KeyFrame---------------------
        first_frame_->SetKeyFrame();
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(first_frame_);
        map_->InsertKeyFrame(current_frame_);

        return true;
    }
} // namespace ECT_SLAM