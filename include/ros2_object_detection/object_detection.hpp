// include/ros2_object_detection/object_detection.hpp
#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

// Standard C++ includes
#include <chrono> // For std::chrono::steady_clock
#include <map>    // For std::map to store tracked objects
#include <mutex>  // For std::mutex
#include <thread> // For std::thread
#include <vector> // For storing object IDs
#include <memory> // For std::unique_ptr
#include <string> // For std::string (used by ROS parameters)

// GStreamer & GLib includes
#include <glib.h>
#include <gst/gst.h>

// ROS 2 includes
#include <rclcpp/rclcpp.hpp>

// ROS 2 message types
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

// DeepStream metadata headers
#include "gstnvdsmeta.h"     // For gst_buffer_get_nvds_batch_meta
#include "nvdsmeta.h"        // For NvDsDisplayMeta and other core metadata structures
#include "nvll_osd_struct.h" // For NvOSD_TextParams, NvOSD_RectParams, NvOSD_LineParams etc.
#include "nvds_tracker_meta.h" // FOR TRACKER_STATE enum and NVDS_TRACKER_METADATA

// Custom local includes
#include "ros2_object_detection/kalman_filter_2d.hpp"
#include "ros2_object_detection/constants.hpp"

// Forward declarations for GStreamer types.
typedef struct _GstElement GstElement;
typedef struct _GstPad GstPad;
typedef struct _GstPadProbeInfo GstPadProbeInfo;
typedef struct _NvDsFrameMeta NvDsFrameMeta;
typedef struct _NvDsBatchMeta NvDsBatchMeta;
typedef struct _NvDsObjectMeta NvDsObjectMeta;

/**
 * @brief Custom constant for an unselected object ID.
 * This is renamed to avoid a conflict with a DeepStream macro.
 */
static const guint64 NO_OBJECT_ID = 0;

/**
 * @brief Custom enum for the tracking status of the selected object.
 * This abstracts the underlying DeepStream and KF states for OSD display.
 */
enum TrackingStatus {
    DETECTED,
    OCCLUDED,
    TRACKED
};

// Forward declaration of OSDRenderer, as it needs to be used in this class.
class OSDRenderer;

/**
 * @class ObjectDetectionNode
 * @brief ROS 2 node for integrating DeepStream with ROS 2.
 *
 * This node manages a GStreamer pipeline for object detection and tracking using DeepStream,
 * publishes detected objects as ROS 2 `Detection2DArray` messages, and publishes the processed
 * video stream as `CompressedImage` messages. It also supports joystick-based target selection
 * and displays custom OSD overlays (FPS, selected target highlight, reticule, velocity).
 */
class ObjectDetectionNode : public rclcpp::Node
{
public:
    /**
     * @brief Constructor for the ObjectDetectionNode.
     * @param options Configuration options for the ROS 2 node.
     */
    explicit ObjectDetectionNode(const rclcpp::NodeOptions &options);

    /**
     * @brief Destructor for the ObjectDetectionNode.
     * Cleans up GStreamer resources and stops threads.
     */
    ~ObjectDetectionNode();

private:
    // --- GStreamer related members ---
    GstElement *pipeline_;           ///< Pointer to the GStreamer pipeline.
    GMainLoop *main_loop_;           ///< Pointer to the GLib main loop for GStreamer.
    std::thread gstreamer_thread_;   ///< Thread to run the GStreamer main loop.

    // --- ROS 2 Publishers ---
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;  ///< Publishes detected objects.
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_publisher_; ///< Publishes compressed video frames.

    // --- ROS 2 Subscriber ---
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_; ///< Subscribes to joystick input.

    // --- Static Callbacks for GStreamer ---
    // These must be static because GStreamer callbacks are C-style function pointers.
    // `user_data` is used to pass a pointer to the `ObjectDetectionNode` instance.
    static GstFlowReturn new_sample_callback(GstElement *sink, gpointer user_data);
    static GstPadProbeReturn osd_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

    // --- Private Methods ---
    /**
     * @brief Callback for processing incoming joystick messages.
     * @param msg The received joystick message.
     */
    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg);

    /**
     * @brief Cycles through detected targets (forward or backward).
     * @param forward If true, cycles to the next target; otherwise, cycles to the previous.
     */
    void cycle_selected_target(bool forward);

    /**
     * @brief Populates a ROS 2 Detection2D message from DeepStream object metadata.
     * @param obj_meta Pointer to the NvDsObjectMeta to process.
     * @param detection_array_msg The ROS 2 Detection2DArray message to populate.
     */
    void populate_ros_detection_message(NvDsObjectMeta *obj_meta, vision_msgs::msg::Detection2DArray &detection_array_msg);

    /**
     * @brief Manages the Kalman Filter for the selected object, including prediction, update, and deselection logic.
     * This method now returns the predicted state and the tracking status.
     * @param selected_object_found_in_frame True if the selected object was detected in the current frame.
     * @param current_selected_bbox_detected The detected bounding box of the selected object, if found.
     * @return The current TrackingStatus of the selected object.
     */
    TrackingStatus manage_selected_object_kalman_filter(
        bool selected_object_found_in_frame,
        const NvOSD_RectParams &current_selected_bbox_detected);
    
    // --- Members for target selection, highlighting, and custom tracking ---
    // Note: These members are now ordered to match the constructor initialization list
    guint64 selected_object_id_;
    std::unique_ptr<KalmanFilter2D> selected_object_kf_;
    bool selected_object_kf_initialized_;
    unsigned int selected_object_lost_frames_;
    bool is_actively_tracking_;
    bool button0_pressed_prev_;
    bool button1_pressed_prev_;
    bool button2_pressed_prev_;
    
    std::map<guint64, NvOSD_RectParams> current_tracked_objects_;
    std::map<guint64, std::string> current_tracked_classes_;
    std::mutex tracked_objects_mutex_;

    NvOSD_RectParams selected_object_last_bbox_;
    std::string selected_object_class_label_;
    
    // OSD Renderer instance
    std::unique_ptr<OSDRenderer> osd_renderer_;

    // Parameters
    std::vector<long int> allowed_class_ids_;
    double camera_fov_rad_;
};

#endif // OBJECT_DETECTION_HPP