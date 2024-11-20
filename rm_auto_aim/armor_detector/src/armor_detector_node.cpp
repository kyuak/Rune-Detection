#include "armor_detector/armor_detector_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>

#include "armor_detector/armor.hpp"

std::chrono::high_resolution_clock::time_point start_time, end_time;
std::chrono::microseconds duration;

namespace rm_auto_aim
{
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions &options) : Node("armor_detector", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting ArmorDetectorNode!");

    // ArmorDetector
    armor_detector_ = initArmorDetector();

    // Visualization Marker Publisher
    // See http://wiki.ros.org/rviz/DisplayTypes/Marker
    armor_marker_.ns       = "armors";
    armor_marker_.action   = visualization_msgs::msg::Marker::ADD;
    armor_marker_.type     = visualization_msgs::msg::Marker::CUBE;
    armor_marker_.scale.x  = 0.05;
    armor_marker_.scale.z  = 0.125;
    armor_marker_.color.a  = 1.0;
    armor_marker_.color.g  = 0.5;
    armor_marker_.color.b  = 1.0;
    armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

    text_marker_.ns       = "classification";
    text_marker_.action   = visualization_msgs::msg::Marker::ADD;
    text_marker_.type     = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker_.scale.z  = 0.1;
    text_marker_.color.a  = 1.0;
    text_marker_.color.r  = 1.0;
    text_marker_.color.g  = 1.0;
    text_marker_.color.b  = 1.0;
    text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/armor_detector/marker", 10);

    // Debug Publishers
    int detector_node_debug_num_ = this->declare_parameter("detector_node_debug", 0);
    detector_node_debug_         = detector_node_debug_num_ ? true : false;

    cam_info_sub_ =
        this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/camera_info",
                                                                rclcpp::SensorDataQoS(),
                                                                [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info)
                                                                {
                                                                    cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
                                                                    cam_info_   = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
                                                                    pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
                                                                    cam_info_sub_.reset();
                                                                });

    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", rclcpp::SensorDataQoS(), std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));

    last_time = now();
}

void ArmorDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
    // count the latency from capturing image to now
    rclcpp::Time reach_time = this->now();
    double latency          = (reach_time - img_msg->header.stamp).seconds() * 1000;
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

    // Detect armors
    std::vector<Armor> armors = detectArmors(img_msg);
}

std::unique_ptr<ArmorDetector> ArmorDetectorNode::initArmorDetector()
{
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    param_desc.integer_range.resize(1);
    param_desc.description                 = "0-RED, 1-BLUE";
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value   = 1;
    auto detect_color                      = declare_parameter("detect_color", BLUE, param_desc);

    param_desc.integer_range[0].step       = 1;
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value   = 255;
    int blue_binary_thres                  = declare_parameter("blue_binary_thres", 50, param_desc);
    int red_binary_thres                   = declare_parameter("red_binary_thres", 100, param_desc);

    ArmorDetector::LightParams l_params = {.min_ratio      = declare_parameter("light.min_ratio", 0.1),
                                           .max_ratio      = declare_parameter("light.max_ratio", 0.4),
                                           .max_angle      = declare_parameter("light.max_angle", 75.0),
                                           .min_fill_ratio = declare_parameter("light.min_fill_ratio", 0.8)};

    ArmorDetector::ArmorParams a_params = {.min_light_ratio           = declare_parameter("armor.min_light_ratio", 0.75),
                                           .min_small_center_distance = declare_parameter("armor.min_small_center_distance", 0.8),
                                           .max_small_center_distance = declare_parameter("armor.max_small_center_distance", 3.2),
                                           .min_large_center_distance = declare_parameter("armor.min_large_center_distance", 3.2),
                                           .max_large_center_distance = declare_parameter("armor.max_large_center_distance", 5.5),
                                           .max_angle                 = declare_parameter("armor.max_angle", 75.0)};

    auto armor_detector =
        std::make_unique<ArmorDetector>(detect_color == RED ? red_binary_thres : blue_binary_thres, detect_color, l_params, a_params);

    // Init classifier
    auto pkg_path                           = ament_index_cpp::get_package_share_directory("armor_detector");
    auto model_path                         = pkg_path + "/model/mlp.onnx";
    auto label_path                         = pkg_path + "/model/label.txt";
    double threshold                        = this->declare_parameter("classifier_threshold", 0.7);
    std::vector<std::string> ignore_classes = this->declare_parameter("ignore_classes", std::vector<std::string>{"negative"});
    armor_detector->classifier              = std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);

    // Init debug mode
    int detector_debug_num_        = this->declare_parameter("detector_debug", 0);
    armor_detector->detector_debug = detector_debug_num_ ? true : false;

    return armor_detector;
}

std::vector<Armor> ArmorDetectorNode::detectArmors(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
{
    // Convert ROS img to cv::Mat
    this->raw_image = cv_bridge::toCvShare(img_msg, "bgr8")->image;

    // Update params
    armor_detector_->detect_color = get_parameter("detect_color").as_int();
    if (armor_detector_->detect_color == RED)
    {
        armor_detector_->binary_thres = get_parameter("red_binary_thres").as_int();
        std::cout << "red_binary_thres: " << armor_detector_->binary_thres << std::endl;
    }

    else
    {
        armor_detector_->binary_thres = get_parameter("blue_binary_thres").as_int();
        std::cout << "blue_binary_thres: " << armor_detector_->binary_thres << std::endl;
    }

    armor_detector_->classifier->threshold = get_parameter("classifier_threshold").as_double();

    std::vector<rm_auto_aim::Armor> armors = armor_detector_->detect(raw_image);

    return armors;
}

void ArmorDetectorNode::publishMarkers()
{
    using Marker         = visualization_msgs::msg::Marker;
    armor_marker_.action = Marker::ADD;
    marker_array_.markers.emplace_back(armor_marker_);
    marker_pub_->publish(marker_array_);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)