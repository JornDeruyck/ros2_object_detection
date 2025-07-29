#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <yaml-cpp/yaml.h>
#include <gst/gst.h>
#include <glib.h>
#include <chrono> // For std::chrono::steady_clock
#include <mutex>  // For std::mutex
#include <thread>
#include "nvdsmeta.h"

class ObjectDetectionNode : public rclcpp::Node
{
public:
    explicit ObjectDetectionNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~ObjectDetectionNode();

    // Method to cycle through detected targets (can be called by joystick input)
    void cycle_selected_target();

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

    // === Members for FPS calculation and display ===
    std::chrono::steady_clock::time_point last_fps_update_time_; // Stores the last time FPS was updated
    unsigned int frame_counter_;                                  // Counts frames since last FPS update
    double current_fps_display_;                                  // Stores the calculated FPS value to be displayed
    std::mutex fps_mutex_;                                        // Mutex to protect FPS variables from concurrent access

    // Members for target selection and highlighting
    guint64 selected_object_id_; // The ID of the currently selected object
    std::map<guint64, NvOSD_RectParams> current_tracked_objects_; // Map of object_id to their last known bounding box
    std::mutex tracked_objects_mutex_; // Mutex to protect current_tracked_objects_ and selected_object_id_
};
