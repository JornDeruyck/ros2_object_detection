#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <yaml-cpp/yaml.h>
#include <gst/gst.h>
#include <thread>

class ObjectDetectionNode : public rclcpp::Node
{
public:
    explicit ObjectDetectionNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~ObjectDetectionNode();

private:
    // === Publishers ===
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_;

    // === GStreamer stuff ===
    GstElement *pipeline_{nullptr};
    GMainLoop *main_loop_{nullptr};
    std::thread gstreamer_thread_;

    // === Callbacks ===
    static GstFlowReturn new_sample_callback(GstElement *sink, gpointer user_data);
    static GstPadProbeReturn osd_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
};
