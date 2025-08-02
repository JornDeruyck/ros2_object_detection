// include/ros2_object_detection/target_logic.hpp
#ifndef TARGET_LOGIC_HPP
#define TARGET_LOGIC_HPP

#include <rclcpp/rclcpp.hpp>
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "geometry_msgs/msg/point.hpp"

#include <string>

/**
 * @class TargetLogicNode
 * @brief Subscribes to detections and a selected target ID to calculate and publish a tracking error.
 *
 * This node represents the core logic for object tracking. It receives all detections
 * from the perception node and a specific target ID to track from a UI or other source.
 * It then finds the target in the detection list, calculates its deviation from the
 * center of the frame, and publishes this as an angular error in radians.
 */
class TargetLogicNode : public rclcpp::Node
{
public:
    /**
     * @brief Constructor for the TargetLogicNode.
     * @param options Configuration options for the ROS 2 node.
     */
    explicit TargetLogicNode(const rclcpp::NodeOptions &options);

private:
    /**
     * @brief Callback for when a new target ID is selected for tracking.
     * @param msg The message containing the ID of the target to track.
     */
    void target_id_callback(const std_msgs::msg::UInt64::SharedPtr msg);

    /**
     * @brief Callback for processing incoming object detections.
     * @param msg The message containing an array of detected objects.
     */
    void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);

    // --- ROS 2 Subscriptions ---
    rclcpp::Subscription<std_msgs::msg::UInt64>::SharedPtr target_id_subscription_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_subscription_;

    // --- ROS 2 Publisher ---
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr tracking_error_publisher_;

    // --- Node State ---
    uint64_t target_id_to_track_; ///< The ID of the object we are currently trying to track.

    // --- Parameters ---
    int frame_width_;          ///< Width of the camera frame in pixels.
    int frame_height_;         ///< Height of the camera frame in pixels.
    double camera_fov_x_rad_;  ///< Horizontal camera field of view in radians.
    double camera_fov_y_rad_;  ///< Vertical camera field of view in radians.
};

#endif // TARGET_LOGIC_HPP
