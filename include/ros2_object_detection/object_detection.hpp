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
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

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

/**
 * @brief Custom enum for the tracking status of the selected object for OSD.
 * This is determined by our logic, not just the tracker's state.
 */
enum class OSDTrackingStatus {
    DETECTED, // Object is actively detected by the tracker.
    OCCLUDED  // Object is lost by the tracker; we are predicting with KF.
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

    // ROS 2 Publishers & Subscribers
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_;
    rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr selected_target_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;

    // Static GStreamer Callbacks
    static GstFlowReturn new_sample_callback(GstElement *sink, gpointer user_data);
    static GstPadProbeReturn osd_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

    // Private Methods
    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);
    void cycle_selected_target(bool forward);
    void populate_ros_detection_message(NvDsObjectMeta *obj_meta, vision_msgs::msg::Detection2DArray &detection_array_msg);
    
    /**
     * @brief Manages the state and KF for the selected object.
     * @param selected_obj_meta Pointer to the selected object's metadata if found, otherwise nullptr.
     * @return The current OSDTrackingStatus.
     */
    OSDTrackingStatus manage_selected_object_state(const NvDsObjectMeta* selected_obj_meta);

    // --- State Members ---
    gint64 selected_object_id_;
    gint64 locked_target_id_;
    
    // --- Custom Kalman Filter Members (Restored) ---
    std::unique_ptr<KalmanFilter2D> selected_object_kf_;
    bool selected_object_kf_initialized_;
    unsigned int selected_object_lost_frames_;
    NvOSD_RectParams selected_object_last_bbox_; // Last known good bbox (from tracker)
    std::string selected_object_class_label_;
    
    // Joystick button state
    bool button0_pressed_prev_;
    bool button1_pressed_prev_;
    bool button2_pressed_prev_;
    
    // Caches for objects detected in the current frame
    std::map<guint64, NvOSD_RectParams> current_tracked_objects_;
    std::map<guint64, std::string> current_tracked_classes_;
    std::mutex tracked_objects_mutex_;
    
    // OSD Renderer
    std::unique_ptr<OSDRenderer> osd_renderer_;

    // Parameters
    std::vector<long int> allowed_class_ids_;
    double camera_fov_rad_;
};

#endif // OBJECT_DETECTION_HPP
