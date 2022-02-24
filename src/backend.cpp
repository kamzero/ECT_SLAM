#include "backend.hpp"
#include "algorithm.hpp"
#include "feature.hpp"
#include "map.hpp"
#include "mappoint.hpp"
#include "g2o_types.hpp"

namespace ECT_SLAM
{

    Backend::Backend()
    {
        backend_running_.store(true);
        backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
    }

    void Backend::UpdateMap()
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.notify_one();
    }

    void Backend::Stop()
    {
        backend_running_.store(false);
        map_update_.notify_one();
        backend_thread_.join();
    }

    void Backend::BackendLoop()
    {
        while (backend_running_.load())
        {
            std::unique_lock<std::mutex> lock(data_mutex_);
            map_update_.wait(lock);

            /// 后端仅优化激活的Frames和Landmarks
            Map::KeyframesType active_kfs = map_->GetAllKeyFrames();
            Map::LandmarksType landmarks = map_->GetAllMapPoints();
            Optimize(active_kfs, landmarks);
        }
    }

    void Backend::Optimize(Map::KeyframesType &keyframes_,
                           Map::LandmarksType &landmarks)
    {
        auto it = keyframes_.begin();
        if(keyframes_.size()>20)
            std::advance(it, keyframes_.size()-20);
        Map::KeyframesType keyframes = Map::KeyframesType(it, keyframes_.end());
        std::cout << "######BackEnd###### Optimizing " << keyframes.size() << std::endl;

        //-----------------setup g2o-----------------
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
            LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        //-----------------paras for vertex & edge-----------------
        // pose 顶点，使用Keyframe id
        std::map<unsigned long, VertexPose *> vertices;
        // 路标顶点，使用路标id索引
        std::map<unsigned long, VertexXYZ *> vertices_landmarks;
        // edges
        int index = 1;
        double chi2_th = 5.991; // robust kernel 阈值
        std::map<EdgeProjection *, Feature::Ptr> edges_and_features;
        // K 和左右外参
        Mat33 K = camera_->K_d();
        SE3 ext = SE3();
        unsigned long max_kf_id = 20;
        int count = 0;

        //-----------------setup vertex & edge-----------------
        for (auto &keyframe : keyframes)
        {
            auto kf = keyframe.second;
            VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
            vertex_pose->setId(count);
            vertex_pose->setEstimate(kf->Pose());
            if(count++ < 2)
                vertex_pose->setFixed(true);
            optimizer.addVertex(vertex_pose);
            vertices.insert({kf->keyframe_id_, vertex_pose});

            for (auto &feat : kf->features_)
            {
                if (feat->is_outlier_ || feat->map_point_.lock() == nullptr)
                    continue;
                auto landmark = feat->map_point_.lock();
                unsigned long landmark_id = landmark->id_;

                // 如果landmark还没有被加入优化，则新加一个顶点
                if (vertices_landmarks.find(landmark_id) ==
                    vertices_landmarks.end())
                {
                    VertexXYZ *v = new VertexXYZ;
                    v->setEstimate(landmark->Pos());
                    v->setId(landmark_id + max_kf_id + 1);
                    v->setMarginalized(true);
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v);
                }

                EdgeProjection *edge = nullptr;
                edge = new EdgeProjection(K, ext);

                edge->setId(index);
                edge->setVertex(0, vertices.at(kf->keyframe_id_));      // pose
                edge->setVertex(1, vertices_landmarks.at(landmark_id)); // landmark
                edge->setMeasurement(toVec2(feat->position_.pt));
                edge->setInformation(Mat22::Identity());
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(chi2_th);
                edge->setRobustKernel(rk);
                edges_and_features.insert({edge, feat});

                optimizer.addEdge(edge);

                index++;
            }
        }

        //-----------------do optimization and eliminate the outliers-----------------
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        int cnt_outlier = 0, cnt_inlier = 0;
        int iteration = 0;
        while (iteration < 5)
        {
            cnt_outlier = 0;
            cnt_inlier = 0;
            // determine if we want to adjust the outlier threshold
            for (auto &ef : edges_and_features)
            {
                if (ef.first->chi2() > chi2_th)
                {
                    cnt_outlier++;
                }
                else
                {
                    cnt_inlier++;
                }
            }
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5)
            {
                break;
            }
            else
            {
                chi2_th *= 2;
                iteration++;
            }
        }

        for (auto &ef : edges_and_features)
        {
            if (ef.first->chi2() > chi2_th)
            {
                ef.second->is_outlier_ = true;
                // remove the observation
                if (auto lock = ef.second->map_point_.lock())
                    lock->RemoveObservation(ef.second);
            }
            else
            {
                ef.second->is_outlier_ = false;
            }
        }

        LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
                  << cnt_inlier;

        //-----------------Set pose and lanrmark position-----------------
        auto iter = keyframes.begin();
        for (auto &v : vertices)
        {
            (iter->second)->SetPose(v.second->estimate());
            iter++;
        }

        int cnt = 0;
        for (auto &v : vertices_landmarks)
        {
            auto iit = landmarks.end();
            if (landmarks.find(v.first) == landmarks.end()){
                cnt++;
            }
            else
                landmarks.at(v.first)->SetPos(v.second->estimate());
        }
        std::cout << vertices_landmarks.size()<< "-vertex " << landmarks.size() << "-landmarks !!!! - not find " << cnt << "\n";
    }

} // namespace ECT_SLAM