#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

#include <rclcpp/rclcpp.hpp>
#include <gst/gst.h>
#include <glib.h>
#include <thread>
#include <chrono> // For std::chrono::steady_clock
#include <mutex>  // For std::mutex
#include <map>    // For std::map to store tracked objects
#include <vector> // For storing object IDs
#include <yaml-cpp/yaml.h> // For YAML parsing (used in .cpp, but good to note dependency)

// DeepStream metadata headers
#include "nvdsmeta.h"     // For NvDsDisplayMeta and other core metadata structures
#include "gstnvdsmeta.h"  // For gst_buffer_get_nvds_batch_meta
#include "nvll_osd_struct.h" // For NvOSD_TextParams, NvOSD_RectParams, NvOSD_LineParams etc.

// ROS messages
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/joy.hpp" // Added for joystick input
#include <opencv2/opencv.hpp> // Not explicitly used for OSD, but good to keep if needed for image processing

// ObjectDetectionNode class inherits from rclcpp::Node
class ObjectDetectionNode : public rclcpp::Node
{
public:
    // Constructor
    explicit ObjectDetectionNode(const rclcpp::NodeOptions &options);
    // Destructor
    ~ObjectDetectionNode();

    // Method to cycle through detected targets (forward or backward)
    void cycle_selected_target(bool forward);

private:
    // GStreamer related members
    GstElement *pipeline_;    // Pointer to the GStreamer pipeline
    GMainLoop *main_loop_;    // Pointer to the GLib main loop for GStreamer
    std::thread gstreamer_thread_; // Thread to run the GStreamer main loop

    // ROS publishers
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_;

    // ROS Joystick Subscriber
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;

    // Static callbacks for GStreamer probes and signals
    // These need to be static because GStreamer callbacks are C-style function pointers,
    // and `user_data` is used to pass a pointer to the `ObjectDetectionNode` instance.
    static GstFlowReturn new_sample_callback(GstElement *sink, gpointer user_data);
    static GstPadProbeReturn osd_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

    // Callback for joystick input
    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);

    // Members for FPS calculation and display
    std::chrono::steady_clock::time_point last_fps_update_time_; // Stores the last time FPS was updated
    unsigned int frame_counter_;                                  // Counts frames since last FPS update
    double current_fps_display_;                                  // Stores the calculated FPS value to be displayed
    std::mutex fps_mutex_;                                        // Mutex to protect FPS variables from concurrent access

    // Members for target selection and highlighting
    guint64 selected_object_id_; // The ID of the currently selected object
    std::map<guint64, NvOSD_RectParams> current_tracked_objects_; // Map of object_id to their last known bounding box
    std::mutex tracked_objects_mutex_; // Mutex to protect current_tracked_objects_ and selected_object_id_

    // New member to track how long the selected object has been lost
    unsigned int selected_object_lost_frames_;

    // To prevent rapid cycling on button hold
    bool button0_pressed_prev_ = false;
    bool button1_pressed_prev_ = false;
};

#endif // OBJECT_DETECTION_HPP
