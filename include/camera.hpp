#pragma once

#include "common_include.hpp"

namespace ECT_SLAM
{

    // Pinhole stereo camera model
    class Camera
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Camera> Ptr;

        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0; // Camera intrinsics
        double k1_ = 0, k2_ = 0, r1_ = 0, r2_ = 0;

        Camera();

        Camera(double fx, double fy, double cx, double cy)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy)
        {
        }

        Mat33f K() const
        {
            Mat33f k;
            k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
            return k;
        }

        // return intrinsic matrix
        Mat33 K_d() const
        {
            Mat33 k;
            k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
            return k;
        }

        // return intrinsic matrix
        cv::Mat K_cv() const
        {
            return (cv::Mat1d(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
        }

        // coordinate transform: world, camera, pixel
        Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);

        Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

        Vec2 camera2pixel(const Vec3 &p_c);

        Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

        Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1);

        Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);
    };

} // namespace ECT_SLAM
