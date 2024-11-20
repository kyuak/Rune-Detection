#include "armor_detector/pnp_solver.hpp"

#include <cstring>
#include <opencv2/calib3d.hpp>
#include <vector>

namespace rm_auto_aim
{
PnPSolver::PnPSolver(
  const std::array<double, 9> & camera_matrix, const std::vector<double> & dist_coeffs)
: camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
  dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone())
{
  // Start from bottom left in clockwise order
  // Model coordinate: x forward, y left, z up
  small_armor_points_.emplace_back(cv::Point3f(0, SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT));
  small_armor_points_.emplace_back(cv::Point3f(0, SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT));
  small_armor_points_.emplace_back(cv::Point3f(0, -SMALL_HALF_WEIGHT, SMALL_HALF_HEIGHT));
  small_armor_points_.emplace_back(cv::Point3f(0, -SMALL_HALF_WEIGHT, -SMALL_HALF_HEIGHT));

  large_armor_points_.emplace_back(cv::Point3f(0, LARGE_HALF_WEIGHT, -LARGE_HALF_HEIGHT));
  large_armor_points_.emplace_back(cv::Point3f(0, LARGE_HALF_WEIGHT, LARGE_HALF_HEIGHT));
  large_armor_points_.emplace_back(cv::Point3f(0, -LARGE_HALF_WEIGHT, LARGE_HALF_HEIGHT));
  large_armor_points_.emplace_back(cv::Point3f(0, -LARGE_HALF_WEIGHT, -LARGE_HALF_HEIGHT));

  // initialize ceres solver
  camera_coeffs[0] = camera_matrix_.at<double>(0, 0);
  camera_coeffs[1] = camera_matrix_.at<double>(1, 1);
  camera_coeffs[2] = camera_matrix_.at<double>(0, 2);
  camera_coeffs[3] = camera_matrix_.at<double>(1, 2);

  ceres::CostFunction * cost_function =
    ReprojectionError_AutoDiff::Create(observations, camera_coeffs);
  problem.AddResidualBlock(cost_function, nullptr /* squared loss */, vecs);

  problem.SetParameterLowerBound(vecs, 0, -1);
  problem.SetParameterUpperBound(vecs, 0, 1);
  problem.SetParameterLowerBound(vecs, 1, -1);
  problem.SetParameterUpperBound(vecs, 1, 1);
  problem.SetParameterLowerBound(vecs, 2, -1);
  problem.SetParameterUpperBound(vecs, 2, 1);

  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.logging_type = ceres::SILENT;
  options.use_nonmonotonic_steps = true;
}

bool PnPSolver::solvePnP_BA(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec)
{
  cur_armor_type = armor.type == ArmorType::SMALL ? 0 : 1;

  rvec.create(3, 1, CV_64F);
  tvec.create(3, 1, CV_64F);

  // Solve pnp
  observations[0] = armor.left_light.bottom.x;
  observations[1] = armor.left_light.bottom.y;
  observations[2] = armor.left_light.top.x;
  observations[3] = armor.left_light.top.y;
  observations[4] = armor.right_light.top.x;
  observations[5] = armor.right_light.top.y;
  observations[6] = armor.right_light.bottom.x;
  observations[7] = armor.right_light.bottom.y;

  // print observations
  // std::cout << "observations: ";
  // for (int i = 0; i < 8; i++) {
  //   std::cout << observations[i] << ", ";
  // }
  // std::cout << std::endl;

  memset(vecs, 0, sizeof(vecs));
  //   vecs[0] = - 15 * CV_PI / 180;
  vecs[5] = 2.0;

  ceres::Solve(options, &problem, &summary);
  //   std::cout << summary.FullReport() << "\n";

  rvec.at<double>(0, 0) = vecs[0];
  rvec.at<double>(1, 0) = vecs[1];
  rvec.at<double>(2, 0) = vecs[2];
  tvec.at<double>(0, 0) = vecs[3];
  tvec.at<double>(1, 0) = vecs[4];
  tvec.at<double>(2, 0) = vecs[5];

  // std::cout << "tvec: " << tvec << std::endl;
  // std::cout << "rvec: " << rvec << std::endl;
  // std::cout << "rvec: " << vecs[0] * 180 / CV_PI << " " << vecs[1] * 180 / CV_PI << " "
  //           << vecs[2] * 180 / CV_PI << " tvec: " << tvec << " cost: " << summary.final_cost
  //           << std::endl;
  return true;
}

bool PnPSolver::solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec)
{
  std::vector<cv::Point2f> image_armor_points;

  // Fill in image points
  image_armor_points.emplace_back(armor.left_light.bottom);
  image_armor_points.emplace_back(armor.left_light.top);
  image_armor_points.emplace_back(armor.right_light.top);
  image_armor_points.emplace_back(armor.right_light.bottom);

  // Solve pnp
  auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
  return cv::solvePnP(
    object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);
}

float PnPSolver::calculateDistanceToCenter(const cv::Point2f & image_point)
{
  float cx = camera_matrix_.at<double>(0, 2);
  float cy = camera_matrix_.at<double>(1, 2);
  return cv::norm(image_point - cv::Point2f(cx, cy));
}

}  // namespace rm_auto_aim