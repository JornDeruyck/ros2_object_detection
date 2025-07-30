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
     * @brief Calculates and updates the FPS display.
     * @param batch_meta Pointer to the current NvDsBatchMeta.
     * @param frame_meta Pointer to the current NvDsFrameMeta.
     */
    void update_and_display_fps(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta);

    /**
     * @brief Populates a ROS 2 Detection2D message from DeepStream object metadata.
     * @param obj_meta Pointer to the NvDsObjectMeta to process.
     * @param detection_array_msg The ROS 2 Detection2DArray message to populate.
     */
    void populate_ros_detection_message(NvDsObjectMeta *obj_meta, vision_msgs::msg::Detection2DArray &detection_array_msg);

    /**
     * @brief Manages the Kalman Filter for the selected object, including prediction, update, and deselection logic.
     * @param selected_object_found_in_frame True if the selected object was detected in the current frame.
     * @param current_selected_bbox_detected The detected bounding box of the selected object, if found.
     * @param current_selected_obj_meta_ptr Pointer to the NvDsObjectMeta of the selected object, if found.
     * @param[out] predicted_x The predicted X coordinate from the KF.
     * @param[out] predicted_y The predicted Y coordinate from the KF.
     * @param[out] predicted_vx The predicted X velocity from the KF.
     * @param[out] predicted_vy The predicted Y velocity from the KF.
     */
    void manage_selected_object_kalman_filter(
        bool selected_object_found_in_frame,
        const NvOSD_RectParams &current_selected_bbox_detected,
        NvDsObjectMeta *current_selected_obj_meta_ptr,
        double &predicted_x, double &predicted_y,
        double &predicted_vx, double &predicted_vy);
    
    /**
     * @brief Draws a reticule (crosshair) at a specified center point on the OSD.
     * @param batch_meta The NvDsBatchMeta for acquiring display metadata.
     * @param frame_meta The NvDsFrameMeta to add display metadata to.
     * @param center_x The X coordinate of the reticule's center.
     * @param center_y The Y coordinate of the reticule's center.
     * @param size The size (length of each arm) of the reticule.
     * @param color The color of the reticule.
     */
    void draw_reticule(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                       gfloat center_x, gfloat center_y, gfloat size, NvOSD_ColorParams color);

    /**
     * @brief Applies OSD customization to a detected object's metadata.
     * This includes setting border color, width, and text label.
     * @param obj_meta The NvDsObjectMeta to modify.
     * @param is_selected_and_kf_initialized True if this is the selected object AND its KF is initialized.
     * @param predicted_vx Predicted X velocity for the selected object (0 for others).
     * @param predicted_vy Predicted Y velocity for the selected object (0 for others).
     * @param selected_object_found_in_frame True if the selected object was detected in the current frame.
     */
    void customize_object_osd(
        NvDsObjectMeta *obj_meta,
        bool is_selected_and_kf_initialized,
        double predicted_vx, double predicted_vy,
        bool selected_object_found_in_frame);

    /**
     * @brief Draws the KF-predicted bounding box, reticule, and associated text for the selected object.
     * This is called regardless of whether the object is currently detected or occluded.
     * @param batch_meta The NvDsBatchMeta for acquiring display metadata.
     * @param frame_meta The NvDsFrameMeta to add display metadata to.
     * @param predicted_vx Predicted X velocity for the selected object.
     * @param predicted_vy Predicted Y velocity for the selected object.
     */
    void draw_selected_object_overlay(
        NvDsBatchMeta *batch_meta,
        NvDsFrameMeta *frame_meta,
        double predicted_vx, double predicted_vy);

    // --- Members for FPS calculation and display ---
    std::chrono::steady_clock::time_point last_fps_update_time_; ///< Stores the last time FPS was updated.
    unsigned int frame_counter_;                                  ///< Counts frames since last FPS update.
    double current_fps_display_;                                  ///< Stores the calculated FPS value to be displayed.
    std::mutex fps_mutex_;                                        ///< Mutex to protect FPS variables from concurrent access.

    // --- Members for target selection, highlighting, and custom tracking ---
    guint64 selected_object_id_;                                  ///< The ID of the currently selected object.
    std::map<guint64, NvOSD_RectParams> current_tracked_objects_; ///< Map of object_id to their last known bounding box (only for currently detected).
    std::mutex tracked_objects_mutex_;                            ///< Mutex to protect current_tracked_objects_ and selected_object_id_.

    // Custom Kalman Filter for the SELECTED object
    std::unique_ptr<KalmanFilter2D> selected_object_kf_; ///< Kalman filter instance for the selected object.
    bool selected_object_kf_initialized_;                 ///< Flag to indicate if the KF for the selected object is initialized.
    unsigned int selected_object_lost_frames_;            ///< Counts frames the selected object has been out of view of the detector (used by KF).

    // New member to store the last known bounding box (predicted or detected) for the selected object
    // This allows drawing the reticule even if the object is occluded.
    NvOSD_RectParams selected_object_last_bbox_;

    // New member to track the DeepStream tracker's reported state for the selected object
    TRACKER_STATE selected_object_tracker_state_;

    // To prevent rapid cycling on button hold
    bool button0_pressed_prev_; ///< Previous state of joystick button 0.
    bool button1_pressed_prev_; ///< Previous state of joystick button 1.
};

#endif // OBJECT_DETECTION_HPP
