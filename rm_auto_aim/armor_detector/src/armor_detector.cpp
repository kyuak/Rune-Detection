#include "armor_detector/armor_detector.hpp"

namespace rm_auto_aim
{
ArmorDetector::ArmorDetector(
  const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a)
: binary_thres(bin_thres), detect_color(color), l(l), a(a)
{
}

std::vector<Armor> ArmorDetector::detect(const cv::Mat & raw_image)
{
  cv::imshow("raw_image", raw_image);
  cv::waitKey(1);

  return armors_;
}

}  // namespace rm_auto_aim