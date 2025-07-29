#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

// Standard C++ includes
#include <chrono> // For std::chrono::steady_clock
#include <map>    // For std::map to store tracked objects
#include <mutex>  // For std::mutex
#include <thread> // For std::thread
#include <vector> // For storing object IDs

// GStreamer & GLib includes
#include <glib.h>
#include <gst/gst.h> // This header already defines GstFlowReturn, GstPadProbeReturn, etc.

// ROS 2 includes
#include <rclcpp/rclcpp.hpp>

// ROS 2 message types
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

// DeepStream metadata headers
// These provide structures and functions for interacting with DeepStream's metadata
#include "gstnvdsmeta.h"     // For gst_buffer_get_nvds_batch_meta
#include "nvdsmeta.h"        // For NvDsDisplayMeta and other core metadata structures
#include "nvll_osd_struct.h" // For NvOSD_TextParams, NvOSD_RectParams, NvOSD_LineParams etc.

// YAML parsing (used in .cpp, but good to note dependency)
#include <yaml-cpp/yaml.h>

// OpenCV (not explicitly used for OSD, but good to keep if needed for image processing elsewhere)
#include <opencv2/opencv.hpp>

// Define a constant for an untracked/unselected object ID.
// Using a large, unlikely ID value to signify "no object selected".
// Make sure this is consistently defined and used.
// A common value is G_MAXUINT64, which is defined in glib.h.
#ifndef UNTRACKED_OBJECT_ID
#define UNTRACKED_OBJECT_ID G_MAXUINT64 // Using G_MAXUINT64 from glib.h for a clear "untracked" state
#endif

// Forward declarations for GStreamer types.
// We only need to forward declare structs/classes, not enums that are already typedef'd.
// GstElement, GstPad, GstPadProbeInfo are structs.
// GstFlowReturn and GstPadProbeReturn are already typedef'd enums by gst.h, so no forward declaration is needed for them.
typedef struct _GstElement GstElement;
typedef struct _GstPad GstPad;
typedef struct _GstPadProbeInfo GstPadProbeInfo;

/**
 * @class ObjectDetectionNode
 * @brief ROS 2 node for integrating DeepStream with ROS 2.
 *
 * This node manages a GStreamer pipeline for object detection and tracking using DeepStream,
 * publishes detected objects as ROS 2 `Detection2DArray` messages, and publishes the processed
 * video stream as `CompressedImage` messages. It also supports joystick-based target selection
 * and displays custom OSD overlays (FPS, selected target highlight, reticule).
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

    /**
     * @brief Cycles through detected targets (forward or backward).
     * @param forward If true, cycles to the next target; otherwise, cycles to the previous.
     */
    void cycle_selected_target(bool forward);

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

    // --- Members for FPS calculation and display ---
    std::chrono::steady_clock::time_point last_fps_update_time_; ///< Stores the last time FPS was updated.
    unsigned int frame_counter_;                                  ///< Counts frames since last FPS update.
    double current_fps_display_;                                  ///< Stores the calculated FPS value to be displayed.
    std::mutex fps_mutex_;                                        ///< Mutex to protect FPS variables from concurrent access.

    // --- Members for target selection and highlighting ---
    guint64 selected_object_id_;                                  ///< The ID of the currently selected object.
    std::map<guint64, NvOSD_RectParams> current_tracked_objects_; ///< Map of object_id to their last known bounding box.
    std::mutex tracked_objects_mutex_;                            ///< Mutex to protect current_tracked_objects_ and selected_object_id_.

    // New member to track how long the selected object has been lost
    unsigned int selected_object_lost_frames_; ///< Counts frames the selected object has been out of view.

    // --- Joystick button state tracking for debouncing ---
    bool button0_pressed_prev_; ///< Previous state of joystick button 0.
    bool button1_pressed_prev_; ///< Previous state of joystick button 1.
};

#endif // OBJECT_DETECTION_HPP