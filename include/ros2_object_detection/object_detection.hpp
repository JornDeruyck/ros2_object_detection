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

// YAML parsing (used in .cpp, but good to note dependency)
#include <yaml-cpp/yaml.h>

// OpenCV (not explicitly used for OSD, but good to keep if needed for image processing elsewhere)
#include <opencv2/opencv.hpp>

// Eigen for Kalman Filter
#include <Eigen/Dense> // For MatrixXd, VectorXd

// Define a constant for an untracked/unselected object ID.
#ifndef UNTRACKED_OBJECT_ID
#define UNTRACKED_OBJECT_ID G_MAXUINT64 // Using G_MAXUINT64 from glib.h for a clear "untracked" state
#endif

// Forward declarations for GStreamer types.
typedef struct _GstElement GstElement;
typedef struct _GstPad GstPad;
typedef struct _GstPadProbeInfo GstPadProbeInfo;

/**
 * @brief Simple 2D Kalman Filter for constant velocity model.
 * State: [x, y, vx, vy]
 * Measurement: [x, y]
 */
class KalmanFilter2D {
public:
    Eigen::MatrixXd A; // State transition matrix
    Eigen::MatrixXd H; // Measurement matrix
    Eigen::MatrixXd Q; // Process noise covariance
    Eigen::MatrixXd R; // Measurement noise covariance
    Eigen::MatrixXd P; // Error covariance matrix
    Eigen::VectorXd x_hat; // State estimate [x, y, vx, vy]

    KalmanFilter2D(); // Constructor initializes matrices

    /**
     * @brief Initializes the filter state and covariance for a new track.
     * @param x Initial x position.
     * @param y Initial y position.
     */
    void init(double x, double y);

    /**
     * @brief Predicts the next state.
     */
    void predict();

    /**
     * @brief Updates the state with a new measurement.
     * @param measured_x Measured x position.
     * @param measured_y Measured y position.
     */
    void update(double measured_x, double measured_y);

    double getX() const { return x_hat(0); }
    double getY() const { return x_hat(1); }
    double getVx() const { return x_hat(2); }
    double getVy() const { return x_hat(3); }
};


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