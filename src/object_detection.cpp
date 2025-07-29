#include "ros2_object_detection/object_detection.hpp"

// Standard C++ includes
#include <algorithm>   // For std::find, std::sort
#include <chrono>      // For FPS calculation
#include <iomanip>     // For std::setprecision
#include <mutex>       // For FPS and object mutexes
#include <sstream>     // For std::stringstream
#include <string>      // For std::string
#include <cmath>       // For std::abs
#include <cstring>     // For memset

// GStreamer & GLib includes
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <glib.h>

// DeepStream metadata headers
#include "nvdsmeta.h"        // For NvDsDisplayMeta and other core metadata structures
#include "gstnvdsmeta.h"     // For gst_buffer_get_nvds_batch_meta
#include "nvll_osd_struct.h" // For NvOSD_TextParams, NvOSD_RectParams, NvOSD_LineParams etc.
#include "nvds_tracker_meta.h" // Crucial for NVDS_TRACKER_METADATA and NvDsTargetMiscDataObject (containing TRACKER_STATE)

// ROS 2 message types
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

// OpenCV (already in header, but good to keep if needed for image processing elsewhere)
#include <opencv2/opencv.hpp>

// Eigen for Kalman Filter
// This is already included in the header via <Eigen/Dense>
// #include <Eigen/Dense>

// Define the threshold for how many frames an object can be lost by our KF before deselection
// This counter is distinct from the DeepStream tracker's internal "lost" state.
#define KF_LOST_THRESHOLD 100 // How many frames our custom Kalman Filter will predict without new measurements before declaring the object lost and deselecting.

#ifndef NVDS_TRACKER_METADATA
#define NVDS_TRACKER_METADATA 103
#endif

// --- KalmanFilter2D Implementation ---
/**
 * @brief Constructor for the KalmanFilter2D class.
 *
 * Initializes the state transition (A), measurement (H), process noise (Q),
 * and measurement noise (R) covariance matrices for a constant velocity 2D Kalman filter.
 */
KalmanFilter2D::KalmanFilter2D() {
    // Resize matrices and vectors
    A.resize(4, 4);
    H.resize(2, 4);
    Q.resize(4, 4);
    R.resize(2, 2);
    P.resize(4, 4);
    x_hat.resize(4);

    // A: State Transition Matrix (Constant Velocity Model)
    // x_k = x_{k-1} + vx_{k-1}*dt
    // y_k = y_{k-1} + vy_{k-1}*dt
    // vx_k = vx_{k-1}
    // vy_k = vy_{k-1}
    // dt: time step between frames. Assuming 30 FPS, dt = 1/30 seconds.
    // This value might need to be adjusted based on the actual frame rate of your video stream.
    double dt = 1.0 / 30.0; // Default: 1/30 sec (for 30 FPS)

    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;

    // H: Measurement Matrix
    // We only measure position (x, y) directly from the object detector.
    // measured_x = x
    // measured_y = y
    H << 1, 0, 0, 0,
         0, 1, 0, 0;

    // Q: Process Noise Covariance Matrix
    // Represents the uncertainty in our state transition model (how much the object's velocity
    // is expected to deviate from a constant model between predictions).
    // Tune these values based on how "smooth" vs. "responsive" you want the filter to be.
    // Higher values make the filter trust its prediction less, leading to faster adaptation but more jitter.
    double process_noise_pos = 1.0;  // Variance in position (pixels^2) per frame for unmodeled acceleration
    double process_noise_vel = 0.5;  // Variance in velocity (pixels/frame)^2 per frame for velocity changes
    Q << process_noise_pos, 0, 0, 0,
         0, process_noise_pos, 0, 0,
         0, 0, process_noise_vel, 0,
         0, 0, 0, process_noise_vel;

    // R: Measurement Noise Covariance Matrix
    // Represents the uncertainty/variance in our measurements (from the object detector).
    // Tune based on the expected accuracy (pixel variance) of your object detector's bounding box centers.
    // Lower values mean the filter trusts new measurements more.
    double measurement_noise_xy = 25.0; // Assuming a variance of 25 pixels^2 (approx. std dev of 5 pixels)
    R << measurement_noise_xy, 0,
         0, measurement_noise_xy;

    // P: Initial Error Covariance Matrix
    // Represents the initial uncertainty in our state estimate.
    // High initial values (e.g., identity matrix scaled by 1000) allow the filter
    // to quickly converge to the first few measurements, effectively trusting them more.
    P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;

    x_hat.setZero(); // Initialize the state estimate to all zeros
}

/**
 * @brief Initializes the Kalman filter state and covariance for a new track.
 * @param x Initial x position (center of bounding box).
 * @param y Initial y position (center of bounding box).
 */
void KalmanFilter2D::init(double x, double y) {
    // Initialize state with detected position and zero initial velocity
    x_hat << x, y, 0, 0;
    // Reset uncertainty to a high value, so the first few updates are heavily weighted by measurements
    P = Eigen::MatrixXd::Identity(4, 4) * 1000.0;
}

/**
 * @brief Predicts the next state of the tracked object.
 *
 * Uses the state transition matrix `A` to project the current state estimate
 * forward in time, and updates the error covariance matrix `P`.
 */
void KalmanFilter2D::predict() {
    x_hat = A * x_hat;
    P = A * P * A.transpose() + Q;
}

/**
 * @brief Updates the state of the tracked object using a new measurement.
 * @param measured_x Measured x position (center of bounding box).
 * @param measured_y Measured y position (center of bounding box).
 *
 * This function calculates the Kalman gain and uses it to correct the predicted state
 * based on the new measurement, reducing uncertainty.
 */
void KalmanFilter2D::update(double measured_x, double measured_y) {
    Eigen::VectorXd z(2); // Measurement vector
    z << measured_x, measured_y;

    Eigen::VectorXd y_tilde = z - H * x_hat;           // Innovation (measurement residual)
    Eigen::MatrixXd S = H * P * H.transpose() + R;     // Innovation covariance
    Eigen::MatrixXd K = P * H.transpose() * S.inverse(); // Kalman gain

    x_hat = x_hat + K * y_tilde;
    P = (Eigen::MatrixXd::Identity(4, 4) - K * H) * P;
}


// --- Appsink callback: get frames, compress, publish ---
/**
 * @brief Static callback function for the GStreamer appsink's "new-sample" signal.
 *
 * This function is invoked when a new video frame (sample) is available from the appsink.
 * It pulls the sample, maps its buffer, and publishes the raw JPEG data as a
 * `sensor_msgs::msg::CompressedImage` ROS 2 message.
 *
 * @param sink The GstElement (appsink) that triggered the callback.
 * @param user_data A pointer to the ObjectDetectionNode instance.
 * @return GstFlowReturn indicating the processing status.
 */
GstFlowReturn ObjectDetectionNode::new_sample_callback(GstElement *sink, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstSample *sample = nullptr;

    // Pull the sample from the appsink
    g_signal_emit_by_name(sink, "pull-sample", &sample);

    if (!sample) {
        RCLCPP_WARN(node->get_logger(), "New sample callback: No sample received.");
        return GST_FLOW_OK;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        RCLCPP_ERROR(node->get_logger(), "New sample callback: Could not get GstBuffer from GstSample.");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR; // Indicate a critical error
    }

    GstMapInfo map;
    // Map the buffer to access its data in read-only mode
    if (gst_buffer_map(buffer, &map, GST_MAP_READ))
    {
        auto msg = sensor_msgs::msg::CompressedImage();
        msg.header.stamp = node->get_clock()->now();
        msg.header.frame_id = "camera_frame"; // Set appropriate frame ID (e.g., from camera_info)
        msg.format = "jpeg";                  // Indicate the image format

        // Efficiently copy the raw JPEG data directly to the message
        msg.data.assign(map.data, map.data + map.size);

        // Publish the compressed image
        node->compressed_publisher_->publish(msg);

        gst_buffer_unmap(buffer, &map);
    }
    else
    {
        RCLCPP_ERROR(node->get_logger(), "New sample callback: Failed to map GstBuffer for reading.");
    }

    // Unreference the sample to release resources
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}


// --- Probe callback: extract detections from metadata and add OSD overlays ---
/**
 * @brief Static probe callback for the sink pad of the nvdsosd element.
 *
 * This function is triggered for each buffer passing through the nvdsosd element.
 * It performs the following:
 * 1. Extracts DeepStream batch metadata to get object detections.
 * 2. Publishes detected objects as `vision_msgs::msg::Detection2DArray`.
 * 3. Calculates and displays FPS as an OSD overlay.
 * 4. Customizes bounding box colors (highlighting selected object).
 * 5. Customizes object labels (including ID and confidence, **custom KF velocity, and DeepStream tracker state**).
 * 6. Manages the state of the `selected_object_id_` (tracking and deselecting if lost by KF and DS tracker).
 * 7. Draws a reticule and predicted bounding box around the selected object (even if occluded).
 *
 * @param pad The GstPad that triggered the probe.
 * @param info Information about the probe, including the GstBuffer.
 * @param user_data A pointer to the ObjectDetectionNode instance.
 * @return GstPadProbeReturn indicating the processing status.
 */
GstPadProbeReturn ObjectDetectionNode::osd_probe_callback(GstPad * /*pad*/, GstPadProbeInfo *info, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstBuffer *gst_buffer = (GstBuffer *)info->data;

    // Retrieve the batch metadata from the GStreamer buffer
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
    if (!batch_meta) {
        RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 1000, "OSD probe: No batch meta found on buffer.");
        return GST_PAD_PROBE_OK;
    }

    // Initialize ROS message for detections
    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header.stamp = node->get_clock()->now();
    detection_array_msg.header.frame_id = "camera_frame"; // Consistent frame ID

    // --- FPS Calculation ---
    std::lock_guard<std::mutex> fps_lock(node->fps_mutex_); // Lock mutex to protect FPS variables during update

    node->frame_counter_++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = now - node->last_fps_update_time_;

    // Update FPS display roughly once per second or every 30 frames
    if (elapsed_seconds.count() >= 1.0 || node->frame_counter_ >= 30)
    {
        node->current_fps_display_ = node->frame_counter_ / elapsed_seconds.count();
        node->frame_counter_ = 0;
        node->last_fps_update_time_ = now;
    }

    // Lock mutex for tracked objects and selected object state before modifying
    std::lock_guard<std::mutex> objects_lock(node->tracked_objects_mutex_);
    node->current_tracked_objects_.clear(); // Clear detected objects from previous frame, will be repopulated

    // Temporary variables to store info about the selected object if detected in this frame
    bool selected_object_found_in_frame = false;
    NvOSD_RectParams current_selected_bbox_detected; // Bounding box from detector if found
    NvDsObjectMeta *current_selected_obj_meta_ptr = nullptr; // Pointer to the actual NvDsObjectMeta if found

    // Iterate through each frame in the batch (typically batch-size=1 for live streams)
    for (GList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) continue; // Skip if frame meta is null

        // --- Add FPS OSD overlay ---
        NvDsDisplayMeta *fps_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (fps_display_meta)
        {
            fps_display_meta->num_labels = 1;
            fps_display_meta->text_params[0].display_text = g_strdup_printf("FPS: %.2f", node->current_fps_display_);
            fps_display_meta->text_params[0].x_offset = 10;
            fps_display_meta->text_params[0].y_offset = 10;
            fps_display_meta->text_params[0].font_params.font_name = (gchar *)"Sans";
            fps_display_meta->text_params[0].font_params.font_size = 14;
            fps_display_meta->text_params[0].font_params.font_color = {1.0, 1.0, 1.0, 1.0}; // RGBA (white)
            fps_display_meta->text_params[0].set_bg_clr = 0;                               // No background box

            nvds_add_display_meta_to_frame(frame_meta, fps_display_meta);
        }

        // --- First pass: Process all detected objects, identify selected one if present ---
        // This loop processes all object detection metadata and populates our internal map.
        // It also identifies if the currently selected object is present in this frame's detections.
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue; // Skip if object meta is null

            node->current_tracked_objects_[obj_meta->object_id] = obj_meta->rect_params; // Store detected bounding box

            // If this object is our currently selected object, record its details
            if (obj_meta->object_id == node->selected_object_id_) {
                selected_object_found_in_frame = true;
                current_selected_bbox_detected = obj_meta->rect_params; // Copy its detected bounding box
                current_selected_obj_meta_ptr = obj_meta; // Store pointer to its NvDsObjectMeta for later modification
            }

            // Populate ROS Detection2D message (done for all detected objects)
            vision_msgs::msg::Detection2D detection;
            detection.header.stamp = detection_array_msg.header.stamp; // Use same timestamp as array header
            detection.header.frame_id = detection_array_msg.header.frame_id; // Consistent frame ID

            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = std::string(obj_meta->obj_label); // Convert gchar* to std::string
            hypothesis.hypothesis.score = obj_meta->confidence;
            detection.results.push_back(hypothesis);

            // Populate bounding box coordinates for ROS message
            detection.bbox.center.position.x = obj_meta->rect_params.left + obj_meta->rect_params.width / 2.0;
            detection.bbox.center.position.y = obj_meta->rect_params.top + obj_meta->rect_params.height / 2.0;
            detection.bbox.size_x = obj_meta->rect_params.width;
            detection.bbox.size_y = obj_meta->rect_params.height;
            detection_array_msg.detections.push_back(detection);

            // Set default OSD parameters for all objects (color will be adjusted in the second pass for selected objects)
            obj_meta->rect_params.border_width = 3;
            obj_meta->rect_params.has_bg_color = 0;
            obj_meta->rect_params.border_color = {0.0, 0.0, 1.0, 1.0}; // Default blue for non-selected objects
        }

        // --- Kalman Filter & Selected Object Tracking Logic ---
        double predicted_x = 0.0, predicted_y = 0.0;
        double predicted_vx = 0.0, predicted_vy = 0.0;

        // Process KF only if an object is currently selected
        if (node->selected_object_id_ != UNTRACKED_OBJECT_ID) {
            // Retrieve DeepStream tracker's state for the selected object from user metadata
            // Reset to EMPTY (lost) for the current frame, then try to find the actual state if object is detected.
            node->selected_object_tracker_state_ = EMPTY;
            if (current_selected_obj_meta_ptr) { // Only attempt to get DS tracker state if selected object was detected in this frame
                NvDsTargetMiscDataObject *selected_tracker_obj_data = nullptr;
                for (GList *l_user_meta = current_selected_obj_meta_ptr->obj_user_meta_list; l_user_meta != nullptr; l_user_meta = l_user_meta->next) {
                    NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user_meta->data;
                    // Check for the specific tracker metadata type.
                    // NVDS_TRACKER_METADATA is defined in nvds_tracker_meta.h
                    if (user_meta->base_meta.meta_type == NVDS_TRACKER_METADATA) {
                        selected_tracker_obj_data = (NvDsTargetMiscDataObject *)user_meta->user_meta_data;
                        if (selected_tracker_obj_data && selected_tracker_obj_data->numObj > 0) {
                            // The `list[0]` typically contains the current frame's tracking info for this object
                            node->selected_object_tracker_state_ = selected_tracker_obj_data->list[0].trackerState;
                        }
                        break;
                    }
                }
            }
            // Note: If the selected object is *not* currently detected (selected_object_found_in_frame is false),
            // its NvDsObjectMeta isn't in frame_meta->obj_meta_list. This means we can't get its `TRACKER_STATE`
            // directly for *this frame*. We'll rely on the `selected_object_lost_frames_` count and
            // our KF's prediction. The `selected_object_tracker_state_` will hold the last known state from when it was detected.

            // Manage Kalman Filter state for the selected object
            if (!node->selected_object_kf_initialized_ || !node->selected_object_kf_) { // If KF is not yet initialized or was reset
                if (selected_object_found_in_frame) {
                    // Initialize KF with the currently detected position and zero initial velocity
                    node->selected_object_kf_ = std::make_unique<KalmanFilter2D>();
                    double center_x = current_selected_bbox_detected.left + current_selected_bbox_detected.width / 2.0;
                    double center_y = current_selected_bbox_detected.top + current_selected_bbox_detected.height / 2.0;
                    node->selected_object_kf_->init(center_x, center_y);
                    node->selected_object_kf_initialized_ = true;
                    node->selected_object_lost_frames_ = 0; // Reset lost counter as it's detected
                    node->selected_object_last_bbox_ = current_selected_bbox_detected; // Store its current detected bbox
                } else {
                    // Selected object not found in this frame, and KF not initialized.
                    // This can happen right after selection if the object is immediately occluded.
                    // It will eventually be deselected by `cycle_selected_target` if no detection occurs.
                    RCLCPP_DEBUG(node->get_logger(), "Selected object %lu not found (DS state: %d) and KF not initialized. Will deselect if not found soon.",
                                 node->selected_object_id_, node->selected_object_tracker_state_);
                }
            } else { // KF is already initialized
                node->selected_object_kf_->predict(); // Always predict first

                if (selected_object_found_in_frame) {
                    // If the selected object is detected, update the KF with this new measurement
                    double center_x = current_selected_bbox_detected.left + current_selected_bbox_detected.width / 2.0;
                    double center_y = current_selected_bbox_detected.top + current_selected_bbox_detected.height / 2.0;
                    node->selected_object_kf_->update(center_x, center_y);
                    node->selected_object_lost_frames_ = 0; // Reset lost counter
                    node->selected_object_last_bbox_ = current_selected_bbox_detected; // Update last known bbox
                } else {
                    // Selected object NOT found in this frame (e.g., occluded by detector, but DS tracker might still track)
                    node->selected_object_lost_frames_++;
                    RCLCPP_DEBUG(node->get_logger(), "Selected object ID %lu KF predicting. Frames lost: %u (DS state: %d)",
                                 node->selected_object_id_, node->selected_object_lost_frames_, node->selected_object_tracker_state_);

                    // Deselection logic for the selected object:
                    // Deselect if our Kalman Filter has been predicting for too long (`KF_LOST_THRESHOLD` frames)
                    // AND the DeepStream tracker also reports the object as `EMPTY` (fully lost/terminated).
                    if (node->selected_object_lost_frames_ > KF_LOST_THRESHOLD && node->selected_object_tracker_state_ == EMPTY)
                    {
                        RCLCPP_INFO(node->get_logger(), "Selected object ID %lu lost by KF (predicted for %u frames) and DeepStream tracker state is EMPTY. Deselecting.",
                                    node->selected_object_id_, node->selected_object_lost_frames_);
                        node->selected_object_id_ = UNTRACKED_OBJECT_ID; // Deselect the object
                        node->selected_object_kf_initialized_ = false;   // Mark KF as uninitialized
                        node->selected_object_kf_.reset();               // Destroy KF instance
                        node->selected_object_lost_frames_ = 0;         // Reset lost frames counter
                        node->selected_object_tracker_state_ = EMPTY;   // Final DS tracker state (lost)
                    } else {
                        // If KF is still within its prediction threshold, or DS tracker is not EMPTY,
                        // continue using KF's prediction for drawing and tracking.
                    }
                }
            } // End of else (KF is initialized)

            // --- Retrieve predicted values ONLY IF KF is still valid AFTER all state updates/deselection ---
            // This is the critical change to prevent null pointer dereference.
            if (node->selected_object_kf_initialized_ && node->selected_object_kf_) {
                predicted_x = node->selected_object_kf_->getX();
                predicted_y = node->selected_object_kf_->getY();
                predicted_vx = node->selected_object_kf_->getVx();
                predicted_vy = node->selected_object_kf_->getVy();

                // Update `selected_object_last_bbox_` with KF's current estimated position for drawing.
                // We keep the original width and height from the last detection to maintain visual size.
                // This will be used for both OSD drawing on `obj_meta` (if detected) or via new display_meta (if occluded).
                node->selected_object_last_bbox_.left = predicted_x - node->selected_object_last_bbox_.width / 2.0;
                node->selected_object_last_bbox_.top = predicted_y - node->selected_object_last_bbox_.height / 2.0;
            } else {
                // If KF is not initialized or was just reset (object deselected),
                // ensure predicted values are zeroed out for OSD display consistency.
                predicted_x = 0.0; predicted_y = 0.0; predicted_vx = 0.0; predicted_vy = 0.0;
                // `selected_object_last_bbox_` will retain its last valid state or its default {0,0,0,0} from constructor
                // when `selected_object_id_` becomes UNTRACKED_OBJECT_ID.
                // If this path is taken, the subsequent drawing logic for the selected object (below)
                // should correctly interpret UNTRACKED_OBJECT_ID and not attempt to draw anything.
            }
        } // End of selected_object_id_ != UNTRACKED_OBJECT_ID block
        else { // selected_object_id_ is UNTRACKED_OBJECT_ID
            // If no object is selected, ensure all KF related states are reset.
            // This handles cases where deselection might happen through `cycle_selected_target` or on node init.
            if (node->selected_object_kf_initialized_ || node->selected_object_kf_) {
                node->selected_object_kf_initialized_ = false;
                node->selected_object_kf_.reset();
                node->selected_object_lost_frames_ = 0;
                node->selected_object_tracker_state_ = EMPTY;
            }
            // Set predicted values to zero as no object is selected.
            predicted_x = 0.0; predicted_y = 0.0; predicted_vx = 0.0; predicted_vy = 0.0;
        }

        // --- Second pass: Customize OSD for all objects, including the selected one ---
        // This loop iterates through all `NvDsObjectMeta` from the DeepStream pipeline.
        // For the selected object, it will override the OSD parameters using our KF's output.
        // For other objects, it uses their default detection metadata.
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue;

            std::stringstream ss_label_conf;

            // Handle OSD for the currently selected object
            if (obj_meta->object_id == node->selected_object_id_)
            {
                // Only use KF predicted info if KF is initialized and an object is still selected
                if (node->selected_object_kf_initialized_ && node->selected_object_id_ != UNTRACKED_OBJECT_ID) {
                    // Override `rect_params` with KF's predicted bounding box for rendering
                    obj_meta->rect_params = node->selected_object_last_bbox_; // Use KF predicted bbox for drawing
                    obj_meta->rect_params.border_color = {1.0, 0.0, 0.0, 1.0}; // Highlight selected object in Red

                    // Build the label text including KF velocity and DS tracker state
                    ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id
                                  << " Conf: " << std::fixed << std::setprecision(2) << obj_meta->confidence; // Keep original confidence

                    // Add KF predicted velocity (Vx, Vy)
                    ss_label_conf << " Vx: " << std::fixed << std::setprecision(2) << predicted_vx
                                  << " Vy: " << std::fixed << std::setprecision(2) << predicted_vy;

                    // Add DeepStream tracker state (e.g., Active, Inactive, Lost)
                    ss_label_conf << " (DS:" << [](TRACKER_STATE state) { // Lambda to convert enum to string
                        switch(state) {
                            case ACTIVE: return "Active";
                            case INACTIVE: return "Inactive";
                            case TENTATIVE: return "Tentative";
                            case PROJECTED: return "Projected";
                            case EMPTY: return "Lost";
                            default: return "Unknown";
                        }
                    }(node->selected_object_tracker_state_) << ")";

                    // Indicate if KF is currently predicting (object not directly detected)
                    if (!selected_object_found_in_frame) {
                        ss_label_conf << " (KF:Pred " << node->selected_object_lost_frames_ << " fr)";
                    }
                } else {
                    // This case occurs if the object was selected, but KF isn't initialized yet (e.g., first frame of selection)
                    // or if it was just deselected and still present in obj_meta_list briefly.
                    // Just draw its detected info with red highlight, without KF/DS state info.
                    ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id
                                  << " Conf: " << std::fixed << std::setprecision(2) << obj_meta->confidence;
                    obj_meta->rect_params.border_color = {1.0, 0.0, 0.0, 1.0}; // Still highlight red
                }

            } else { // Handle OSD for all other (non-selected) objects
                ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id
                              << " Conf: " << std::fixed << std::setprecision(2) << obj_meta->confidence;
                // For non-selected objects, keep their default blue border color
                obj_meta->rect_params.border_color = {0.0, 0.0, 1.0, 1.0};
            }

            std::string label_conf_str = ss_label_conf.str();

            // Free previous display_text and set the new one
            if (obj_meta->text_params.display_text)
            {
                g_free(obj_meta->text_params.display_text);
                obj_meta->text_params.display_text = nullptr;
            }
            obj_meta->text_params.display_text = g_strdup(label_conf_str.c_str());

            // Position text relative to the bounding box (either detected or KF-predicted)
            obj_meta->text_params.x_offset = (guint)obj_meta->rect_params.left;
            gint preferred_y_signed = (gint)(obj_meta->rect_params.top - 25);
            if (preferred_y_signed < 0) { // If text would go off-screen upwards, push it below the box
                obj_meta->text_params.y_offset = (guint)(obj_meta->rect_params.top + obj_meta->rect_params.height + 5);
            } else {
                obj_meta->text_params.y_offset = (guint)preferred_y_signed;
            }

            obj_meta->text_params.font_params.font_name = (gchar *)"Sans";
            obj_meta->text_params.font_params.font_size = 12;
            obj_meta->text_params.font_params.font_color = obj_meta->rect_params.border_color; // Match color to bbox
            obj_meta->text_params.set_bg_clr = 0;                                               // No background box
        } // End of second pass object loop

        // --- Draw Reticule and OSD Text for Selected Object (even if occluded and not in obj_meta_list) ---
        // This is crucial: If the selected object is *not* currently detected by the primary detector (i.e., it's occluded),
        // its `NvDsObjectMeta` won't be in `frame_meta->obj_meta_list`. In this case, we need to acquire new
        // `NvDsDisplayMeta` to manually draw its predicted bounding box, reticule, and KF text.
        // This block only executes if an object is selected, KF is initialized, AND it was NOT found by detector in this frame.
        if (node->selected_object_id_ != UNTRACKED_OBJECT_ID && node->selected_object_kf_initialized_ && !selected_object_found_in_frame)
        {
            // Use KF's predicted bounding box for drawing (from `selected_object_last_bbox_`)
            NvOSD_RectParams &selected_rect_for_drawing = node->selected_object_last_bbox_;

            gfloat center_x = selected_rect_for_drawing.left + selected_rect_for_drawing.width / 2.0;
            gfloat center_y = selected_rect_for_drawing.top + selected_rect_for_drawing.height / 2.0;
            gfloat reticule_size = 20.0;

            // Acquire NvDsDisplayMeta for the reticule (crosshair)
            NvDsDisplayMeta *reticule_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            if (reticule_display_meta)
            {
                reticule_display_meta->num_lines = 2; // Two lines for crosshair
                NvOSD_ColorParams reticule_color = {1.0, 0.0, 0.0, 1.0}; // Red color for selected object's reticule

                // Horizontal line of the reticule
                reticule_display_meta->line_params[0].x1 = (guint)(center_x - reticule_size / 2.0);
                reticule_display_meta->line_params[0].y1 = (guint)center_y;
                reticule_display_meta->line_params[0].x2 = (guint)(center_x + reticule_size / 2.0);
                reticule_display_meta->line_params[0].y2 = (guint)center_y;
                reticule_display_meta->line_params[0].line_width = 2;
                reticule_display_meta->line_params[0].line_color = reticule_color;

                // Vertical line of the reticule
                reticule_display_meta->line_params[1].x1 = (guint)center_x;
                reticule_display_meta->line_params[1].y1 = (guint)(center_y - reticule_size / 2.0);
                reticule_display_meta->line_params[1].x2 = (guint)center_x;
                reticule_display_meta->line_params[1].y2 = (guint)(center_y + reticule_size / 2.0);
                reticule_display_meta->line_params[1].line_width = 2;
                reticule_display_meta->line_params[1].line_color = reticule_color;

                nvds_add_display_meta_to_frame(frame_meta, reticule_display_meta);
            }

            // Acquire NvDsDisplayMeta for the predicted bounding box
            NvDsDisplayMeta *predicted_bbox_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            if (predicted_bbox_display_meta) {
                predicted_bbox_display_meta->num_rects = 1;
                predicted_bbox_display_meta->rect_params[0] = selected_rect_for_drawing; // Use KF predicted bbox
                predicted_bbox_display_meta->rect_params[0].border_color = {1.0, 0.0, 0.0, 1.0}; // Red
                predicted_bbox_display_meta->rect_params[0].border_width = 3;
                predicted_bbox_display_meta->rect_params[0].has_bg_color = 0; // No background fill
                nvds_add_display_meta_to_frame(frame_meta, predicted_bbox_display_meta);
            }


            // Acquire NvDsDisplayMeta for the KF prediction text label
            NvDsDisplayMeta *kf_text_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            if (kf_text_display_meta) {
                kf_text_display_meta->num_labels = 1;
                std::stringstream kf_text_ss;
                kf_text_ss << "ID: " << node->selected_object_id_
                           << " Vx: " << std::fixed << std::setprecision(2) << predicted_vx
                           << " Vy: " << std::fixed << std::setprecision(2) << predicted_vy
                           << " (DS:" << [](TRACKER_STATE state) { // Lambda for DS state string
                                switch(state) {
                                    case ACTIVE: return "Active";
                                    case INACTIVE: return "Inactive";
                                    case TENTATIVE: return "Tentative";
                                    case PROJECTED: return "Projected";
                                    case EMPTY: return "Lost";
                                    default: return "Unknown";
                                }
                            }(node->selected_object_tracker_state_) << ")"
                           << " (KF:Pred " << node->selected_object_lost_frames_ << " fr)"; // KF prediction status

                kf_text_display_meta->text_params[0].display_text = g_strdup(kf_text_ss.str().c_str());
                kf_text_display_meta->text_params[0].x_offset = (guint)(selected_rect_for_drawing.left);
                // Position text above the predicted bounding box, adjust if off-screen
                gint text_y_offset_signed = (gint)(selected_rect_for_drawing.top - 25);
                if (text_y_offset_signed < 0) {
                    kf_text_display_meta->text_params[0].y_offset = (guint)(selected_rect_for_drawing.top + selected_rect_for_drawing.height + 5);
                } else {
                    kf_text_display_meta->text_params[0].y_offset = (guint)text_y_offset_signed;
                }
                kf_text_display_meta->text_params[0].font_params.font_name = (gchar *)"Sans";
                kf_text_display_meta->text_params[0].font_params.font_size = 12;
                kf_text_display_meta->text_params[0].font_params.font_color = {1.0, 1.0, 1.0, 1.0}; // White text for KF info
                kf_text_display_meta->text_params[0].set_bg_clr = 0;
                nvds_add_display_meta_to_frame(frame_meta, kf_text_display_meta);
            }
        }
    } // End of frame loop


    // Publish the array of detections if any objects were detected
    if (!detection_array_msg.detections.empty())
    {
        node->detection_publisher_->publish(detection_array_msg);
    }

    return GST_PAD_PROBE_OK;
}

// --- Constructor ---
/**
 * @brief Initializes the ObjectDetectionNode.
 *
 * This constructor performs the following setup:
 * 1. Initializes ROS 2 parameters and publishers/subscribers.
 * 2. Loads the GStreamer pipeline string from a YAML configuration file.
 * 3. Initializes GStreamer and creates the main loop.
 * 4. Parses and sets up the GStreamer pipeline.
 * 5. Attaches the `osd_probe_callback` to the `nvdsosd` element's sink pad
 * to process DeepStream metadata and add custom OSD.
 * 6. Attaches the `new_sample_callback` to the `appsink` element's "new-sample" signal
 * to retrieve processed video frames.
 * 7. Starts the GStreamer pipeline and runs the GLib main loop in a separate thread.
 *
 * @param options Configuration options for the ROS 2 node.
 */
ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions &options)
    : Node("object_detection_node", options),
      last_fps_update_time_(std::chrono::steady_clock::now()), // Initialize FPS timer
      frame_counter_(0),                                       // Initialize frame counter
      current_fps_display_(0.0),                               // Initialize displayed FPS
      selected_object_id_(UNTRACKED_OBJECT_ID),                // Initialize selected object to untracked
      selected_object_kf_initialized_(false),                  // Initialize KF state to not initialized
      selected_object_lost_frames_(0),                         // Initialize KF lost frames counter
      selected_object_tracker_state_(EMPTY),                   // Initialize DeepStream tracker state to EMPTY
      button0_pressed_prev_(false),                            // Initialize button 0 state for debouncing
      button1_pressed_prev_(false)                             // Initialize button 1 state for debouncing
{
    // Initialize selected_object_last_bbox_ using traditional aggregate initialization for C++17 compatibility.
    // All members are initialized in declaration order.
    NvOSD_RectParams selected_object_last_bbox_ = {
        0.0f, 0.0f, 0.0f, 0.0f,            // left, top, width, height
        0,                                 // border_width
        {0.0f, 0.0f, 0.0f, 0.0f},          // border_color (NvOSD_ColorParams)
        0,                                 // has_bg_color
        0,                                 // reserved (was missing!)
        {0.0f, 0.0f, 0.0f, 0.0f},          // bg_color (NvOSD_ColorParams)
        0,                                 // has_color_info
        0                                  // color_id
    };

    selected_object_last_bbox_.reserved = 0;

    // Declare and load pipeline configuration from ROS parameter
    this->declare_parameter<std::string>("pipeline_config", ""); // Declare with an empty default
    auto config_path = this->get_parameter("pipeline_config").as_string();

    if (config_path.empty()) {
        RCLCPP_FATAL(this->get_logger(), "Parameter 'pipeline_config' is not set. Please provide a path to the YAML config file.");
        throw std::runtime_error("Missing 'pipeline_config' parameter.");
    }

    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::BadFile &e) {
        RCLCPP_FATAL(this->get_logger(), "Failed to load YAML config file '%s': %s", config_path.c_str(), e.what());
        throw; // Re-throw to terminate node startup
    } catch (const YAML::Exception &e) {
        RCLCPP_FATAL(this->get_logger(), "Error parsing YAML config file '%s': %s", config_path.c_str(), e.what());
        throw; // Re-throw to terminate node startup
    }

    std::string pipeline_str = config["pipeline"].as<std::string>();
    if (pipeline_str.empty()) {
        RCLCPP_FATAL(this->get_logger(), "GStreamer pipeline string is empty in config file '%s'.", config_path.c_str());
        throw std::runtime_error("Empty GStreamer pipeline string.");
    }

    // Define QoS settings for publishers to minimize buffering and latency
    rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(1)); // Keep only the latest message
    qos_profile.reliable();                                     // Ensure delivery (vs. best_effort)
    qos_profile.durability_volatile();                          // Messages are not persistent

    // Create ROS publishers for detections and compressed images
    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", qos_profile);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", qos_profile);

    // Create joystick subscriber
    // IMPORTANT: You need a separate ROS2 node (e.g., joy_node or keyboard_joy_node) publishing to /joy
    // Check your joystick driver's output for actual button mappings.
    joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", 10, std::bind(&ObjectDetectionNode::joy_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribing to /joy topic for joystick input.");


    // Init GStreamer
    gst_init(nullptr, nullptr);
    main_loop_ = g_main_loop_new(nullptr, FALSE); // Create a new GLib main loop

    GError *error = nullptr;

    // Parse the GStreamer pipeline string from YAML
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline_)
    {
        RCLCPP_FATAL(this->get_logger(), "Failed to parse GStreamer pipeline: %s", error ? error->message : "Unknown error");
        if (error) g_error_free(error);
        return; // Exit constructor gracefully on pipeline parse failure
    }

    // Hook probe to the sink pad of 'nvdsosd_0'
    // This ensures that all inference and tracking metadata is available,
    // and the custom OSD will be rendered by nvdsosd.
    GstElement *osd_element = gst_bin_get_by_name(GST_BIN(pipeline_), "nvdsosd_0");
    if (osd_element)
    {
        GstPad *osd_sink_pad = gst_element_get_static_pad(osd_element, "sink");
        if (osd_sink_pad) {
            // Add probe with `this` (pointer to ObjectDetectionNode instance) as user_data
            gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_probe_callback, this, nullptr);
            gst_object_unref(osd_sink_pad);
            RCLCPP_INFO(this->get_logger(), "Attached OSD probe to 'nvdsosd_0' sink pad.");
        } else {
            RCLCPP_WARN(this->get_logger(), "Could not get sink pad from 'nvdsosd_0' element for probe attachment. Custom OSD will not function.");
            // If OSD element exists but pad doesn't, it's a configuration issue. Pipeline can still run without OSD customization.
        }
        gst_object_unref(osd_element); // Always unref elements obtained by gst_bin_get_by_name
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find 'nvdsosd_0' element in pipeline. Custom OSD will not be rendered.");
        // Continue without probe if nvdsosd is not found, but custom OSD won't work
    }

    // Hook appsink for frames: Connect the new_sample_callback to the "new-sample" signal of the appsink.
    // This allows us to retrieve processed frames for publishing.
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline_), "ros_sink");
    if (appsink)
    {
        g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample_callback), this);
        gst_object_unref(appsink);
        RCLCPP_INFO(this->get_logger(), "Attached new sample callback to 'ros_sink' appsink.");
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find 'ros_sink' appsink in pipeline. Compressed image publication will not function.");
    }

    // Start the GStreamer pipeline
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    RCLCPP_INFO(this->get_logger(), "Starting GStreamer main loop...");
    // Run the GLib main loop in a separate thread to avoid blocking the ROS2 spin
    gstreamer_thread_ = std::thread([this]() { g_main_loop_run(main_loop_); });
}

// --- Destructor ---
/**
 * @brief Destructor for the ObjectDetectionNode.
 *
 * This destructor ensures proper shutdown and cleanup of all resources:
 * 1. Quits the GLib main loop and joins the GStreamer thread.
 * 2. Sets the GStreamer pipeline state to NULL and unreferences it.
 * 3. Unreferences the GLib main loop.
 */
ObjectDetectionNode::~ObjectDetectionNode()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down ObjectDetectionNode...");

    // 1. Quit the GLib main loop and join the GStreamer thread
    if (main_loop_ && g_main_loop_is_running(main_loop_))
    {
        RCLCPP_INFO(this->get_logger(), "Quitting GLib main loop...");
        g_main_loop_quit(main_loop_);
    }
    // Join the GStreamer thread to ensure it finishes before resources are deallocated
    if (gstreamer_thread_.joinable())
    {
        RCLCPP_INFO(this->get_logger(), "Joining GStreamer thread...");
        gstreamer_thread_.join();
    }
    RCLCPP_INFO(this->get_logger(), "GStreamer thread joined.");

    // 2. Set pipeline state to NULL and unreference it to release all GStreamer resources
    if (pipeline_)
    {
        RCLCPP_INFO(this->get_logger(), "Setting GStreamer pipeline to NULL state and unreferencing...");
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr; // Clear pointer after unref to prevent dangling pointer
    }

    // 3. Unreference the GLib main loop
    if (main_loop_)
    {
        RCLCPP_INFO(this->get_logger(), "Unreferencing GLib main loop...");
        g_main_loop_unref(main_loop_);
        main_loop_ = nullptr; // Clear pointer after unref
    }

    RCLCPP_INFO(this->get_logger(), "ObjectDetectionNode shut down complete.");
}

// --- Method to cycle through detected targets ---
/**
 * @brief Cycles the `selected_object_id_` to the next or previous detected object.
 *
 * This method is called in response to joystick input. It iterates through the
 * `current_tracked_objects_` to find the next/previous object ID based on the
 * `forward` parameter. Object IDs are sorted to ensure consistent cycling order.
 * If no object is currently selected or the current selection is no longer valid,
 * it selects the first one (or last if cycling backward).
 *
 * @param forward If true, cycles to the next object; otherwise, cycles to the previous.
 */
void ObjectDetectionNode::cycle_selected_target(bool forward)
{
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);

    if (current_tracked_objects_.empty())
    {
        selected_object_id_ = UNTRACKED_OBJECT_ID; // No objects to select
        // Also reset KF if no objects are detected at all
        selected_object_kf_initialized_ = false;
        selected_object_kf_.reset(); // Destroy KF instance if no objects to track
        RCLCPP_INFO(this->get_logger(), "No objects currently detected to select.");
        return;
    }

    // Get all current object IDs and sort them to ensure consistent cycling order
    std::vector<guint64> object_ids;
    for (const auto& pair : current_tracked_objects_)
    {
        object_ids.push_back(pair.first);
    }
    std::sort(object_ids.begin(), object_ids.end());

    // Find the currently selected object's position in the sorted vector
    auto it = std::find(object_ids.begin(), object_ids.end(), selected_object_id_);

    if (it == object_ids.end() || selected_object_id_ == UNTRACKED_OBJECT_ID)
    {
        // If no object is selected (or current selection is no longer valid),
        // select the first one (or last if cycling backward).
        if (forward) {
            selected_object_id_ = object_ids[0];
        } else {
            selected_object_id_ = object_ids.back();
        }
    }
    else
    {
        if (forward) {
            // Move to the next object, or wrap around to the first if at the end
            ++it;
            if (it == object_ids.end())
            {
                selected_object_id_ = object_ids[0]; // Wrap around to first object
            }
            else
            {
                selected_object_id_ = *it;
            }
        } else {
            // Move to the previous object, or wrap around to the last if at the beginning
            if (it == object_ids.begin())
            {
                selected_object_id_ = object_ids.back(); // Wrap around to last object
            }
            else
            {
                --it;
                selected_object_id_ = *it;
            }
        }
    }
    RCLCPP_INFO(this->get_logger(), "New selected object ID: %lu", selected_object_id_);
    // When a new object is selected, re-initialize its Kalman Filter related states
    selected_object_kf_initialized_ = false; // KF will be initialized on next detection
    selected_object_kf_.reset();             // Destroy any previous KF instance
    selected_object_lost_frames_ = 0;        // Reset KF lost frames counter
    selected_object_tracker_state_ = EMPTY;  // Reset DeepStream tracker state for new selection
}

// --- Callback for joystick input ---
/**
 * @brief Processes incoming `sensor_msgs::msg::Joy` messages to handle target cycling.
 *
 * This callback implements simple debouncing for joystick buttons 0 (forward) and 1 (backward).
 * It triggers the `cycle_selected_target` method only on the rising edge of a button press.
 *
 * @param msg The received joystick message.
 */
void ObjectDetectionNode::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    // Define joystick button indices. Adjust these based on your specific joystick mapping.
    const int FORWARD_BUTTON_INDEX = 0;  // Example: 'A' button on Xbox controller
    const int BACKWARD_BUTTON_INDEX = 1; // Example: 'B' button on Xbox controller

    // Check if button indices are valid and get current button states
    bool current_button0_pressed = (msg->buttons.size() > FORWARD_BUTTON_INDEX && msg->buttons[FORWARD_BUTTON_INDEX] == 1);
    bool current_button1_pressed = (msg->buttons.size() > BACKWARD_BUTTON_INDEX && msg->buttons[BACKWARD_BUTTON_INDEX] == 1);

    // Debounce for forward button: trigger only on press (rising edge)
    if (current_button0_pressed && !button0_pressed_prev_)
    {
        RCLCPP_INFO(this->get_logger(), "Joystick button %d pressed. Cycling target forward.", FORWARD_BUTTON_INDEX);
        cycle_selected_target(true); // Cycle forward
    }
    // Debounce for backward button: trigger only on press (rising edge)
    else if (current_button1_pressed && !button1_pressed_prev_)
    {
        RCLCPP_INFO(this->get_logger(), "Joystick button %d pressed. Cycling target backward.", BACKWARD_BUTTON_INDEX);
        cycle_selected_target(false); // Cycle backward
    }

    // Update previous button states for the next callback to enable debouncing
    button0_pressed_prev_ = current_button0_pressed;
    button1_pressed_prev_ = current_button1_pressed;
}

// --- Main entrypoint ---
/**
 * @brief Main function for the ROS 2 ObjectDetectionNode executable.
 *
 * This function initializes the ROS 2 system, creates an instance of the
 * `ObjectDetectionNode`, and starts spinning the node to process callbacks.
 * It includes a try-catch block for robust error handling during node startup.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return An integer representing the program's exit status.
 */
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv); // Initialize ROS 2

    rclcpp::NodeOptions options;
    // Parameters can be passed via ROS launch files or command line:
    // e.g., `ros2 run ros2_object_detection object_detection_node --ros-args -p pipeline_config:=/path/to/your_pipeline_config.yaml`

    // Create and run the ObjectDetectionNode. Use a shared_ptr for automatic memory management.
    std::shared_ptr<ObjectDetectionNode> node = nullptr;
    try {
        node = std::make_shared<ObjectDetectionNode>(options);
        rclcpp::spin(node); // Spin the ROS 2 node to process callbacks and keep it alive
    } catch (const std::exception& e) {
        // Catch standard exceptions during node initialization or spin
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Node initialization or runtime error: %s", e.what());
    } catch (...) {
        // Catch any other unknown exceptions
        RCLCPP_ERROR(rclcpp::get_logger("main"), "An unknown error occurred during node execution.");
    }

    rclcpp::shutdown(); // Shutdown ROS 2 system
    return 0;
}