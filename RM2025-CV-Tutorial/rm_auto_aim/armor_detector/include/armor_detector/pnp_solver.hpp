#ifndef ARMOR_DETECTOR__PNP_SOLVER_HPP_
#define ARMOR_DETECTOR__PNP_SOLVER_HPP_

// OpenCV
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core.hpp>

// Ceres
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// STD
#include <array>
#include <vector>

#include "armor_detector/armor.hpp"

namespace rm_auto_aim
{
// 0 for small armor, 1 for large armor
static bool cur_armor_type = 0;

class PnPSolver
{
public:
  PnPSolver(
    const std::array<double, 9> & camera_matrix,
    const std::vector<double> & distortion_coefficients);

  // Get 3d position
  bool solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec);
  bool solvePnP_BA(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec);

  // Calculate the distance between armor center and image center
  float calculateDistanceToCenter(const cv::Point2f & image_point);

private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;

  // ceres coefficients
  // rotation vector and translation vector
  double vecs[6];
  // The camera is parameterized using 4 parameters: 2 for focal length and 2 for center.
  double camera_coeffs[4], observations[8];

  // Regularization coefficients
  static constexpr double WEIGHT = 0.1;
  static constexpr double CENTER = -15 * CV_PI / 180;

  // Unit: mm
  static constexpr float SMALL_ARMOR_WIDTH = 135;
  static constexpr float SMALL_ARMOR_HEIGHT = 55;
  static constexpr float LARGE_ARMOR_WIDTH = 225;
  static constexpr float LARGE_ARMOR_HEIGHT = 55;
  static constexpr double SMALL_HALF_WEIGHT = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  static constexpr double SMALL_HALF_HEIGHT = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  static constexpr double LARGE_HALF_WEIGHT = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  static constexpr double LARGE_HALF_HEIGHT = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // Four vertices of armor in 3d
  std::vector<cv::Point3f> small_armor_points_;
  std::vector<cv::Point3f> large_armor_points_;

  // Ceres solver
  ceres::Problem problem;
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;

  // Ceres solver using auto diff
  struct ReprojectionError_AutoDiff
  {
    ReprojectionError_AutoDiff(double * observed, double * camera_coeffs)
    : observed(observed), camera_coeffs(camera_coeffs)
    {
    }

    template <typename T>
    bool operator()(const T * const vecs, T * residuals) const
    {
      T p[12];
      T points_SMALL[12] = {T(-SMALL_HALF_WEIGHT), T(SMALL_HALF_HEIGHT),  T(0),
                            T(-SMALL_HALF_WEIGHT), T(-SMALL_HALF_HEIGHT), T(0),
                            T(SMALL_HALF_WEIGHT),  T(-SMALL_HALF_HEIGHT), T(0),
                            T(SMALL_HALF_WEIGHT),  T(SMALL_HALF_HEIGHT),  T(0)};

      T points_LARGE[12] = {T(-LARGE_HALF_WEIGHT), T(LARGE_HALF_HEIGHT),  T(0),
                            T(-LARGE_HALF_WEIGHT), T(-LARGE_HALF_HEIGHT), T(0),
                            T(LARGE_HALF_WEIGHT),  T(-LARGE_HALF_HEIGHT), T(0),
                            T(LARGE_HALF_WEIGHT),  T(LARGE_HALF_HEIGHT),  T(0)};

      for (int i = 0; i < 4; i++) {
        if (cur_armor_type) {
          ceres::AngleAxisRotatePoint(vecs, points_LARGE + 3 * i, p + 3 * i);
        } else {
          ceres::AngleAxisRotatePoint(vecs, points_SMALL + 3 * i, p + 3 * i);
        }
      }

      const double &focal_x = camera_coeffs[0], &focal_y = camera_coeffs[1];
      const double &center_x = camera_coeffs[2], &center_y = camera_coeffs[3];

      // camera[3,4,5] are the translation.
      for (int i = 0; i < 4; i++) {
        p[0 + 3 * i] += vecs[3];
        p[1 + 3 * i] += vecs[4];
        p[2 + 3 * i] += vecs[5];

        T xp = p[0 + 3 * i] / p[2 + 3 * i];
        T yp = p[1 + 3 * i] / p[2 + 3 * i];

        T predicted_x = focal_x * xp + center_x;
        T predicted_y = focal_y * yp + center_y;

        residuals[0 + 2 * i] = predicted_x - observed[0 + 2 * i];
        residuals[1 + 2 * i] = predicted_y - observed[1 + 2 * i];
      }
      // Regularization
      residuals[8] = WEIGHT * (vecs[0] - CENTER) * (vecs[0] - CENTER);

      return true;
    }

    static ceres::CostFunction * Create(double * observed, double * camera_coeffs)
    {
      return (new ceres::AutoDiffCostFunction<ReprojectionError_AutoDiff, 9, 6>(
        new ReprojectionError_AutoDiff(observed, camera_coeffs)));
    }

    double * observed;
    double * camera_coeffs;
  };
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__PNP_SOLVER_HPP_