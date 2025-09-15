// include/ros2_object_detection/object_detection.hpp
#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <memory>
#include <string>

#include <glib.h>
#include <gst/gst.h>

#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/compressed_image.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvll_osd_struct.h"
#include "nvds_tracker_meta.h"

#include "ros2_object_detection/kalman_filter_2d.hpp"
#include "ros2_object_detection/constants.hpp"

class OSDRenderer;

enum class OSDTrackingStatus {
    DETECTED,
    OCCLUDED
};

struct TrackedObjectState {
    guint64 id;
    std::string class_label;
    float confidence;
    NvOSD_RectParams last_bbox;
    std::shared_ptr<KalmanFilter2D> kf;
    unsigned int frames_since_seen;
    bool kf_initialized;

    TrackedObjectState() : id(0), confidence(0.0), frames_since_seen(0), kf_initialized(false) {}
};

class ObjectDetectionNode : public rclcpp::Node
{
public:
    explicit ObjectDetectionNode(const rclcpp::NodeOptions &options);
    ~ObjectDetectionNode();

private:
    // ... GStreamer and ROS members ...
    GstElement *pipeline_;
    GMainLoop *main_loop_;
    std::thread gstreamer_thread_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
    rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr target_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr lock_target_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr unlock_target_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cycle_target_forward_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cycle_target_backward_service_;

    // Static Callbacks
    static GstFlowReturn new_sample_callback(GstElement *sink, gpointer user_data);
    static GstPadProbeReturn osd_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    static gboolean bus_callback(GstBus *bus, GstMessage *msg, gpointer data);
    
    // Core Logic and Handlers
    void add_latency_probes(GstBin *bin);
    static GstPadProbeReturn latency_probe_sink(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    static GstPadProbeReturn latency_probe_src(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    static void element_added_callback(GstBin *bin, GstElement *element, gpointer user_data);
    void calculate_and_clean_latency(GstBuffer *gst_buffer);
    void update_tracking_state(NvDsBatchMeta* batch_meta);
    void prune_lost_tracks();
    void render_osd(NvDsBatchMeta* batch_meta, const std::vector<TrackedObjectState>& objects_to_render, std::optional<guint64> locked_id, std::optional<guint64> selected_id);
    void publish_messages(const std::vector<TrackedObjectState>& objects_to_render, std::optional<guint64> locked_id, const rclcpp::Time& stamp);
    void handle_lock_target(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void handle_unlock_target(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void handle_cycle_forward(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void handle_cycle_backward(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void handle_bus_message(GstMessage *msg);
    void cycle_selected_target(bool forward);
    void populate_ros_detection_message(const TrackedObjectState& object_state, vision_msgs::msg::Detection2D& detection_msg, const rclcpp::Time& stamp);
    
    // --- State Members ---
    std::optional<guint64> selected_object_id_;
    std::optional<guint64> locked_target_id_;
    
    std::map<guint64, TrackedObjectState> persistent_object_map_;
    std::mutex tracked_objects_mutex_;

    // ... Latency, OSD, and Parameter members ...
    std::map<GstBuffer*, std::map<std::string, std::chrono::steady_clock::time_point>> latency_map_;
    std::map<std::string, double> smoothed_latency_map_;
    std::mutex latency_mutex_;
    std::unique_ptr<OSDRenderer> osd_renderer_;
    std::string frame_id_;
    std::string osd_element_name_;
    std::string appsink_element_name_;
    std::vector<long int> allowed_class_ids_;
    double camera_fov_rad_;
    unsigned int kf_lost_threshold_;
    double latency_smoothing_alpha_;
    bool enable_latency_measurement_;
};

#endif // OBJECT_DETECTION_HPP