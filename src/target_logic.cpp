// src/target_logic.cpp
#include "ros2_object_detection/target_logic.hpp"
#include <string>
#include <cmath> // For M_PI, std::tan, std::atan

// --- Constructor ---
TargetLogicNode::TargetLogicNode(const rclcpp::NodeOptions &options)
    : Node("target_logic_node", options), target_id_to_track_(0)
{
    // --- Parameters ---
    this->declare_parameter<std::string>("detection_topic", "/detections");
    this->declare_parameter<std::string>("selected_target_topic", "/selected_target_id");
    this->declare_parameter<std::string>("tracking_error_topic", "/tracking_error");
    this->declare_parameter<int>("frame_width", 1280);
    this->declare_parameter<int>("frame_height", 720);
    this->declare_parameter<double>("camera_fov", 90.0); // Horizontal FOV in degrees

    auto detection_topic = this->get_parameter("detection_topic").as_string();
    auto selected_target_topic = this->get_parameter("selected_target_topic").as_string();
    auto tracking_error_topic = this->get_parameter("tracking_error_topic").as_string();
    frame_width_ = this->get_parameter("frame_width").as_int();
    frame_height_ = this->get_parameter("frame_height").as_int();
    double camera_fov_deg = this->get_parameter("camera_fov").as_double();

    // --- Convert FOV to radians and calculate vertical FOV ---
    camera_fov_x_rad_ = camera_fov_deg * M_PI / 180.0;
    if (frame_height_ > 0 && frame_width_ > 0) {
        double aspect_ratio = static_cast<double>(frame_width_) / frame_height_;
        camera_fov_y_rad_ = 2.0 * std::atan(std::tan(camera_fov_x_rad_ / 2.0) / aspect_ratio);
    } else {
        camera_fov_y_rad_ = 0.0; // Avoid division by zero
    }

    RCLCPP_INFO(this->get_logger(), "Target Logic Node initialized.");
    RCLCPP_INFO(this->get_logger(), "  Frame Size: %dx%d", frame_width_, frame_height_);
    RCLCPP_INFO(this->get_logger(), "  Horiz. FOV: %.2f deg (%.4f rad)", camera_fov_deg, camera_fov_x_rad_);
    RCLCPP_INFO(this->get_logger(), "  Vert. FOV: %.4f rad", camera_fov_y_rad_);


    // --- QoS Profile ---
    rclcpp::QoS qos_profile(rclcpp::KeepLast(10));
    qos_profile.reliable();

    // --- Subscriptions ---
    target_id_subscription_ = this->create_subscription<std_msgs::msg::Int64>(
        selected_target_topic, qos_profile,
        std::bind(&TargetLogicNode::target_id_callback, this, std::placeholders::_1));

    detection_subscription_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
        detection_topic, qos_profile,
        std::bind(&TargetLogicNode::detection_callback, this, std::placeholders::_1));

    // --- Publisher ---
    tracking_error_publisher_ = this->create_publisher<geometry_msgs::msg::Point>(tracking_error_topic, qos_profile);

    // --- Service clients ---
    tracking_client_ = this->create_client<std_srvs::srv::Trigger>("/pat_controller/operational/tracking");
    automatic_client_ = this->create_client<std_srvs::srv::Trigger>("/pat_controller/operational/automatic");

}

// --- Callbacks ---

void TargetLogicNode::target_id_callback(const std_msgs::msg::Int64::SharedPtr msg)
{
    if (msg->data != target_id_to_track_)
    {
        target_id_to_track_ = msg->data;

        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();

        if (target_id_to_track_ < 0)
        {
            RCLCPP_INFO(this->get_logger(), "Tracking disabled (received target ID -1).");

            geometry_msgs::msg::Point error_msg;
            error_msg.x = 0.0;
            error_msg.y = 0.0;
            error_msg.z = 0.0;
            tracking_error_publisher_->publish(error_msg);

            if (automatic_client_->wait_for_service(std::chrono::seconds(1)))
            {
                automatic_client_->async_send_request(
                    request,
                    [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
                        auto response = future.get();
                        if (response->success)
                        {
                            RCLCPP_INFO(this->get_logger(), "Switched to automatic mode: %s", response->message.c_str());
                        }
                        else
                        {
                            RCLCPP_WARN(this->get_logger(), "Failed to switch to automatic mode: %s", response->message.c_str());
                        }
                    });
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Automatic service not available.");
            }
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "New target to track with ID: %lu", target_id_to_track_);

            if (tracking_client_->wait_for_service(std::chrono::seconds(1)))
            {
                tracking_client_->async_send_request(
                    request,
                    [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
                        auto response = future.get();
                        if (response->success)
                        {
                            RCLCPP_INFO(this->get_logger(), "Switched to tracking mode: %s", response->message.c_str());
                        }
                        else
                        {
                            RCLCPP_WARN(this->get_logger(), "Failed to switch to tracking mode: %s", response->message.c_str());
                        }
                    });
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Tracking service not available.");
            }
        }
    }
}

void TargetLogicNode::detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
    if (target_id_to_track_ < 0) {
        return;
    }

    bool target_found = false;
    for (const auto &detection : msg->detections)
    {
        int64_t current_detection_id = -1;
        try {
            current_detection_id = std::stoull(detection.id);
        } catch (const std::exception&) {
            continue;
        }

        if (current_detection_id == target_id_to_track_)
        {
            target_found = true;

            // Calculate the pixel error from the center of the frame
            double error_x_px = (frame_width_ / 2.0) - detection.bbox.center.position.x;
            double error_y_px = (frame_height_ / 2.0) - detection.bbox.center.position.y;

            // --- Convert pixel error to angular error in radians ---
            auto error_msg = geometry_msgs::msg::Point();
            error_msg.x = (error_x_px / frame_width_) * camera_fov_x_rad_;
            error_msg.y = (error_y_px / frame_height_) * camera_fov_y_rad_;
            error_msg.z = 0.0; // z is unused

            tracking_error_publisher_->publish(error_msg);
            break;
        }
    }

    if (!target_found)
    {
        auto error_msg = geometry_msgs::msg::Point();
        error_msg.x = 0.0;
        error_msg.y = 0.0;
        error_msg.z = 0.0;
        tracking_error_publisher_->publish(error_msg);
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Target ID %ld not found in current detections.", target_id_to_track_);
    }
}

// --- Main entrypoint ---
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<TargetLogicNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
