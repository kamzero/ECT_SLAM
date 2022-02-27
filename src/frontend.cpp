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
        ratio_for_keyframe_ = Config::Get<double>("ratio_for_keyframe");
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

    bool Frontend::Track()
    {
        std::cout << "Tracking No." << current_frame_->id_ << " ... ";

        DetectFeature();
        // initial guess
        current_frame_->SetPose(last_frame_->Pose());
        // current_frame_->SetPose(relative_motion_ * last_frame_->Pose());

        //!--------------PnP Estimate With 2D-3D Matches(map)--------------------------
        std::vector<cv::DMatch> matches;
        std::vector<cv::Point2d> points_2d;
        std::vector<cv::Point3d> points_3d;

        //! TODO: deal with failure
        if (!MatchWith3DMap(matches, points_3d, points_2d))
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

        //!--------------Add New MapPoints With 2D-2D Matches(last frame)--------------
        MatchAndUpdateMap(last_frame_, current_frame_);

        // end stage
        status_ = FrontendStatus::TRACKING_GOOD;
        // relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();
        if (viewer_)
            viewer_->AddCurrentFrame(current_frame_);

        return true;
    }

    bool Frontend::Reset()
    {
        std::cout << "Reset is not implemented. " << std::endl;
        return true;
    }

    bool Frontend::MatchWith3DMap(std::vector<cv::DMatch> &matches,
                                  std::vector<cv::Point3d> &points_3d, std::vector<cv::Point2d> &points_2d)
    {
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        BfMatch3D(active_landmarks, current_frame_->descriptors_, matches);

        std::cout << "MatchWith3DMap " << matches.size() << " / " << active_landmarks.size() << std::endl;
        if (matches.size() < 5)
            return false;
        for (auto m : matches)
        {
            auto iter = active_landmarks.find(m.queryIdx);
            auto pt3d = iter->second->Pos();
            points_3d.emplace_back(pt3d[0], pt3d[1], pt3d[2]);
            points_2d.emplace_back(current_frame_->features_[m.trainIdx]->position_.pt.x, current_frame_->features_[m.trainIdx]->position_.pt.y);

            // Add Obs
            iter->second->AddObservation(current_frame_->features_[m.trainIdx]);
            current_frame_->features_[m.trainIdx]->status_ = STATUS::MATCH3D;
            current_frame_->features_[m.trainIdx]->map_point_ = iter->second;
        }

        return true;
    }

    bool Frontend::DetectFeature()
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(current_frame_->img_, keypoints, 40);
        ComputeORB(current_frame_->img_, keypoints, current_frame_->descriptors_);

        for (int i = 0; i < keypoints.size(); i++)
        {
            Feature::Ptr feature(new Feature(current_frame_, keypoints[i], current_frame_->descriptors_[i]));
            current_frame_->features_.push_back(feature);
        }

        return true;
    }

    bool Frontend::Match2D2D(Frame::Ptr &frame1, Frame::Ptr frame2,
                             std::vector<cv::DMatch> &matches,
                             std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, int thres = 8)
    {
        BfMatch(frame1->descriptors_, frame2->descriptors_, matches);
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
            // if (points3F.at<cv::Vec3f>(i, 0)[2] <= 0) continue;

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
        // std::cout << "-----se3-----\n"<< pose.log() << std::endl;
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

    bool Frontend::MatchAndUpdateMap(Frame::Ptr frame1, Frame::Ptr frame2)
    {
        std::vector<cv::DMatch> matches;
        std::vector<cv::Point2f> points1, points2;
        //!-----------------------Match----------------------------
        if (!Match2D2D(frame1, frame2, matches, points1, points2, 3))
            return false;

        //!-----------------------Delete Matches of MapPoints---------------
        int i = 0;
        std::vector<cv::DMatch> new_matches;
        std::vector<cv::Point2f> new_points1, new_points2;
        for (auto iter = matches.begin(); iter != matches.end(); i++)
        {
            if (!frame1->features_[iter->queryIdx]->map_point_.lock())
            {
                new_matches.push_back(matches[i]);
                new_points1.push_back(points1[i]);
                new_points2.push_back(points2[i]);
            }
            ++iter;
        }
        //!-----------------------Trangulation & Build Map From 2D-2D Matches----------------------------
        Trangulation(frame1, frame2, new_matches, new_points1, new_points2);

        double ratio = (double)matches.size() / (double)frame2->features_.size();
        if (ratio < ratio_for_keyframe_)
        {
            current_frame_->SetKeyFrame();
            map_->InsertKeyFrame(current_frame_);
            backend_->UpdateMap();
            if (viewer_)
                viewer_->UpdateMap();
        }
        return true;
    }

    // brute-force matching
    void BfMatch3D(const Map::LandmarksType &landmarks, const vector<DescType> &desc, vector<cv::DMatch> &matches)
    {
        const int d_max = 40;
        for (auto iter = landmarks.begin(); iter != landmarks.end(); iter++)
        {
            auto i1 = iter->first;
            std::list<std::weak_ptr<Feature>> obs = iter->second->GetObs();

            if (obs.empty())
                continue;
            bool flag = false;
            cv::DMatch m{i1, 0, 256};

            for (auto ob : obs)
            {
                auto lock = ob.lock();
                if (lock)
                    flag = true;
                else
                    continue;

                for (size_t i2 = 0; i2 < desc.size(); ++i2)
                {
                    if (desc[i2].empty())
                        continue;

                    int distance = 0;
                    for (int k = 0; k < 8; k++)
                    {
                        distance += _mm_popcnt_u32(lock->descriptor_[k] ^ desc[i2][k]);
                    }
                    if (distance < d_max && distance < m.distance)
                    {
                        m.distance = distance;
                        m.trainIdx = i2;
                    }
                }
            }

            if (flag && m.distance < d_max)
            {
                matches.push_back(m);
            }
        }
    }

    // void BfMatch3D(const Map::LandmarksType &landmarks, const vector<DescType> &desc, vector<cv::DMatch> &matches)
    // {
    //     const int d_max = 40;
    //     // space for time
    //     std::vector<std::list<std::weak_ptr<Feature>>> landmark_list;
    //     std::vector<unsigned long> landmark_id;
    //     for (auto iter = landmarks.begin(); iter != landmarks.end(); iter++)
    //     {
    //         landmark_list.push_back(iter->second->GetObs());
    //         landmark_id.push_back(iter->first);
    //     }

    //     for (size_t i2 = 0; i2 < desc.size(); ++i2)
    //     {
    //         if (desc[i2].empty())
    //             continue;
    //         cv::DMatch m{0, i2, 256};

    //         auto it = landmark_id.begin();
    //         for (auto iter = landmark_list.begin(); iter != landmark_list.end(); it++, iter++)
    //         {
    //             auto i1 = *it;
    //             std::list<std::weak_ptr<Feature>> obs = *iter;

    //             if (obs.empty())
    //                 continue;
    //             bool flag = false;

    //             for (auto ob : obs)
    //             {
    //                 auto lock = ob.lock();
    //                 if (lock)
    //                     flag = true;
    //                 else
    //                     continue;

    //                 int distance = 0;
    //                 for (int k = 0; k < 8; k++)
    //                 {
    //                     distance += _mm_popcnt_u32(lock->descriptor_[k] ^ desc[i2][k]);
    //                 }
    //                 if (distance < d_max && distance < m.distance)
    //                 {
    //                     m.distance = distance;
    //                     m.queryIdx = i1;
    //                 }
    //             }

    //             if (flag && m.distance < d_max)
    //             {
    //                 matches.push_back(m);
    //             }
    //         }
    //     }
    // }
} // namespace ECT_SLAM