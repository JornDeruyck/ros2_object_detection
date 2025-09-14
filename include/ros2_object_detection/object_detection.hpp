// include/ros2_object_detection/object_detection.hpp
#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

#include <chrono>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <memory>
#include <string>

#include <glib.h>
#include <gst/gst.h>

#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "std_srvs/srv/trigger.hpp"

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvll_osd_struct.h"
#include "nvds_tracker_meta.h"

// Custom local includes
#include "ros2_object_detection/kalman_filter_2d.hpp"
#include "ros2_object_detection/constants.hpp"

// Forward declarations
typedef struct _GstElement GstElement;
typedef struct _GstPad GstPad;
typedef struct _GstPadProbeInfo GstPadProbeInfo;
typedef struct _NvDsFrameMeta NvDsFrameMeta;
typedef struct _NvDsBatchMeta NvDsBatchMeta;
typedef struct _NvDsObjectMeta NvDsObjectMeta;

static const gint64 NO_OBJECT_ID = -1;

enum class OSDTrackingStatus {
    DETECTED,
    OCCLUDED
};

// Struct to hold the complete state of a tracked object
struct TrackedObjectState {
    guint64 id;
    std::string class_label;
    float confidence;
    NvOSD_RectParams last_bbox;
    std::unique_ptr<KalmanFilter2D> kf;
    unsigned int frames_since_seen;
    bool kf_initialized;

    TrackedObjectState() : id(0), confidence(0.0), frames_since_seen(0), kf_initialized(false) {}
};

class OSDRenderer;

class ObjectDetectionNode : public rclcpp::Node
{
public:
    explicit ObjectDetectionNode(const rclcpp::NodeOptions &options);
    ~ObjectDetectionNode();

private:
    // GStreamer members
    GstElement *pipeline_;
    GMainLoop *main_loop_;
    std::thread gstreamer_thread_;

    // ROS 2 Publishers
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
    rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr target_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_;
    rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr selected_target_publisher_;

    // ROS 2 Services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr lock_target_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr unlock_target_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cycle_target_forward_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cycle_target_backward_service_;

    // Static GStreamer Callbacks
    static GstFlowReturn new_sample_callback(GstElement *sink, gpointer user_data);
    static GstPadProbeReturn osd_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    
    // --- Latency Measurement Callbacks & Helpers ---
    void add_latency_probes(GstBin *bin);
    static GstPadProbeReturn latency_probe_sink(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    static GstPadProbeReturn latency_probe_src(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    static void element_added_callback(GstBin *bin, GstElement *element, gpointer user_data);

    // Private Methods
    void cycle_selected_target(bool forward);
    void populate_ros_detection_message(const TrackedObjectState& object_state, vision_msgs::msg::Detection2D& detection_msg, const rclcpp::Time& stamp);
    void handle_lock_target(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void handle_unlock_target(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void handle_cycle_forward(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void handle_cycle_backward(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    
    // --- State Members ---
    gint64 selected_object_id_;
    gint64 locked_target_id_;
    
    // --- New Persistent Tracking Map ---
    std::map<guint64, TrackedObjectState> persistent_object_map_;
    std::mutex tracked_objects_mutex_;

    // --- Latency Measurement ---
    std::map<GstBuffer*, std::map<std::string, std::chrono::steady_clock::time_point>> latency_map_;
    std::map<std::string, double> smoothed_latency_map_;
    
    // OSD Renderer
    std::unique_ptr<OSDRenderer> osd_renderer_;

    // Parameters
    std::vector<long int> allowed_class_ids_;
    double camera_fov_rad_;
};

#endif // OBJECT_DETECTION_HPP