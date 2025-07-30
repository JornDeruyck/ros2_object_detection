// src/object_detection.cpp
#include "ros2_object_detection/object_detection.hpp"
#include "ros2_object_detection/kalman_filter_2d.hpp" // Included for KalmanFilter2D use
#include "ros2_object_detection/constants.hpp"        // Included for UNTRACKED_OBJECT_ID, KF_LOST_THRESHOLD

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

// YAML parsing
#include <yaml-cpp/yaml.h>

// OpenCV (not explicitly used for OSD, but good to keep if needed for image processing elsewhere)
#include <opencv2/opencv.hpp>

// --- Appsink callback: get frames, compress, publish ---
GstFlowReturn ObjectDetectionNode::new_sample_callback(GstElement *sink, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstSample *sample = nullptr;

    g_signal_emit_by_name(sink, "pull-sample", &sample);

    if (!sample) {
        RCLCPP_WARN(node->get_logger(), "New sample callback: No sample received.");
        return GST_FLOW_OK;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        RCLCPP_ERROR(node->get_logger(), "New sample callback: Could not get GstBuffer from GstSample.");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ))
    {
        auto msg = sensor_msgs::msg::CompressedImage();
        msg.header.stamp = node->get_clock()->now();
        msg.header.frame_id = "camera_frame";
        msg.format = "jpeg";
        msg.data.assign(map.data, map.data + map.size);
        node->compressed_publisher_->publish(msg);
        gst_buffer_unmap(buffer, &map);
    }
    else
    {
        RCLCPP_ERROR(node->get_logger(), "New sample callback: Failed to map GstBuffer for reading.");
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// --- Probe callback: extract detections from metadata and add OSD overlays ---
GstPadProbeReturn ObjectDetectionNode::osd_probe_callback(GstPad * /*pad*/, GstPadProbeInfo *info, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstBuffer *gst_buffer = (GstBuffer *)info->data;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
    if (!batch_meta) {
        RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 1000, "OSD probe: No batch meta found on buffer.");
        return GST_PAD_PROBE_OK;
    }

    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header.stamp = node->get_clock()->now();
    detection_array_msg.header.frame_id = "camera_frame";

    // Iterate through each frame in the batch (typically batch-size=1 for live streams)
    for (GList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) continue;

        // Call with corrected arguments: batch_meta and frame_meta
        node->update_and_display_fps(batch_meta, frame_meta); // CORRECTED CALL

        std::lock_guard<std::mutex> objects_lock(node->tracked_objects_mutex_);
        node->current_tracked_objects_.clear(); // Clear detected objects from previous frame

        bool selected_object_found_in_frame = false;
        NvOSD_RectParams current_selected_bbox_detected = {}; // Initialize with default values
        NvDsObjectMeta *current_selected_obj_meta_ptr = nullptr;

        // First pass: Process all detected objects, identify selected one if present
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue;

            node->current_tracked_objects_[obj_meta->object_id] = obj_meta->rect_params;

            if (obj_meta->object_id == node->selected_object_id_) {
                selected_object_found_in_frame = true;
                current_selected_bbox_detected = obj_meta->rect_params;
                current_selected_obj_meta_ptr = obj_meta;
            }
            // Populate ROS Detection2D message (done for all detected objects)
            vision_msgs::msg::Detection2D detection;
            detection.header.stamp = detection_array_msg.header.stamp;
            detection.header.frame_id = detection_array_msg.header.frame_id;

            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = std::string(obj_meta->obj_label);
            hypothesis.hypothesis.score = obj_meta->confidence;
            detection.results.push_back(hypothesis);

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

        double predicted_x = 0.0, predicted_y = 0.0;
        double predicted_vx = 0.0, predicted_vy = 0.0;

        node->manage_selected_object_kalman_filter(
            selected_object_found_in_frame,
            current_selected_bbox_detected,
            current_selected_obj_meta_ptr,
            predicted_x, predicted_y, predicted_vx, predicted_vy);

        // Second pass: Customize OSD for all objects, including the selected one
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue;

            bool is_selected = (obj_meta->object_id == node->selected_object_id_ && node->selected_object_kf_initialized_);
            node->customize_object_osd(
                obj_meta,
                is_selected,
                predicted_vx, predicted_vy,
                selected_object_found_in_frame);
        }

        // Draw reticule and OSD text for selected object (even if occluded)
        if (node->selected_object_id_ != UNTRACKED_OBJECT_ID && node->selected_object_kf_initialized_ && !selected_object_found_in_frame)
        {
            node->draw_selected_object_overlay_if_occluded(
                batch_meta, frame_meta, predicted_vx, predicted_vy);
        }
    } // End of frame loop

    if (!detection_array_msg.detections.empty())
    {
        node->detection_publisher_->publish(detection_array_msg);
    }

    return GST_PAD_PROBE_OK;
}

// --- Constructor ---
ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions &options)
    : Node("object_detection_node", options),
      last_fps_update_time_(std::chrono::steady_clock::now()),
      frame_counter_(0),
      current_fps_display_(0.0),
      selected_object_id_(UNTRACKED_OBJECT_ID),
      selected_object_kf_initialized_(false),
      selected_object_lost_frames_(0),
      selected_object_tracker_state_(EMPTY),
      button0_pressed_prev_(false),
      button1_pressed_prev_(false)
{
    // Initialize selected_object_last_bbox_ with zero values
    memset(&selected_object_last_bbox_, 0, sizeof(NvOSD_RectParams));

    this->declare_parameter<std::string>("pipeline_config", "");
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
        throw;
    } catch (const YAML::Exception &e) {
        RCLCPP_FATAL(this->get_logger(), "Error parsing YAML config file '%s': %s", config_path.c_str(), e.what());
        throw;
    }

    std::string pipeline_str = config["pipeline"].as<std::string>();
    if (pipeline_str.empty()) {
        RCLCPP_FATAL(this->get_logger(), "GStreamer pipeline string is empty in config file '%s'.", config_path.c_str());
        throw std::runtime_error("Empty GStreamer pipeline string.");
    }

    rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();

    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", qos_profile);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", qos_profile);

    joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", 10, std::bind(&ObjectDetectionNode::joy_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribing to /joy topic for joystick input.");

    gst_init(nullptr, nullptr);
    main_loop_ = g_main_loop_new(nullptr, FALSE);

    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline_)
    {
        RCLCPP_FATAL(this->get_logger(), "Failed to parse GStreamer pipeline: %s", error ? error->message : "Unknown error");
        if (error) g_error_free(error);
        throw std::runtime_error("GStreamer pipeline parsing failed.");
    }

    GstElement *osd_element = gst_bin_get_by_name(GST_BIN(pipeline_), "nvdsosd_0");
    if (osd_element)
    {
        GstPad *osd_sink_pad = gst_element_get_static_pad(osd_element, "sink");
        if (osd_sink_pad) {
            gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_probe_callback, this, nullptr);
            gst_object_unref(osd_sink_pad);
            RCLCPP_INFO(this->get_logger(), "Attached OSD probe to 'nvdsosd_0' sink pad.");
        } else {
            RCLCPP_WARN(this->get_logger(), "Could not get sink pad from 'nvdsosd_0' element for probe attachment. Custom OSD will not function.");
        }
        gst_object_unref(osd_element);
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find 'nvdsosd_0' element in pipeline. Custom OSD will not be rendered.");
    }

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

    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    RCLCPP_INFO(this->get_logger(), "Starting GStreamer main loop...");
    gstreamer_thread_ = std::thread([this]() { g_main_loop_run(main_loop_); });
}

// --- Destructor ---
ObjectDetectionNode::~ObjectDetectionNode()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down ObjectDetectionNode...");

    if (main_loop_ && g_main_loop_is_running(main_loop_))
    {
        RCLCPP_INFO(this->get_logger(), "Quitting GLib main loop...");
        g_main_loop_quit(main_loop_);
    }
    if (gstreamer_thread_.joinable())
    {
        RCLCPP_INFO(this->get_logger(), "Joining GStreamer thread...");
        gstreamer_thread_.join();
    }
    RCLCPP_INFO(this->get_logger(), "GStreamer thread joined.");

    if (pipeline_)
    {
        RCLCPP_INFO(this->get_logger(), "Setting GStreamer pipeline to NULL state and unreferencing...");
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }

    if (main_loop_)
    {
        RCLCPP_INFO(this->get_logger(), "Unreferencing GLib main loop...");
        g_main_loop_unref(main_loop_);
        main_loop_ = nullptr;
    }

    RCLCPP_INFO(this->get_logger(), "ObjectDetectionNode shut down complete.");
}

// --- Private Methods for OSD and Logic Splitting ---

void ObjectDetectionNode::update_and_display_fps(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta)
{
    std::lock_guard<std::mutex> fps_lock(fps_mutex_);

    frame_counter_++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = now - last_fps_update_time_;

    if (elapsed_seconds.count() >= 1.0 || frame_counter_ >= 30)
    {
        current_fps_display_ = frame_counter_ / elapsed_seconds.count();
        frame_counter_ = 0;
        last_fps_update_time_ = now;
    }

    // Now correctly passing batch_meta
    NvDsDisplayMeta *fps_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (fps_display_meta)
    {
        fps_display_meta->num_labels = 1;
        fps_display_meta->text_params[0].display_text = g_strdup_printf("FPS: %.2f", current_fps_display_);
        fps_display_meta->text_params[0].x_offset = 10;
        fps_display_meta->text_params[0].y_offset = 10;
        fps_display_meta->text_params[0].font_params.font_name = (gchar *)"Sans";
        fps_display_meta->text_params[0].font_params.font_size = 14;
        fps_display_meta->text_params[0].font_params.font_color = {1.0, 1.0, 1.0, 1.0}; // RGBA (white)
        fps_display_meta->text_params[0].set_bg_clr = 0;
        nvds_add_display_meta_to_frame(frame_meta, fps_display_meta);
    }
}

// NOTE: This function's return value (bool) and output parameters need careful handling
// within osd_probe_callback if you intend to directly modify `selected_object_found_in_frame`,
// `current_selected_bbox_detected`, and `current_selected_obj_meta_ptr`.
// For simplicity in this refactor, `process_object_meta` now only handles the ROS message population
// and internal tracking map. The `selected_object_found_in_frame` and related variable
// will still be managed directly within `osd_probe_callback`'s loop to avoid complex returns/references.
// This function name `process_object_meta` is kept for the ROS message part.
/*
bool ObjectDetectionNode::process_object_meta(NvDsObjectMeta *obj_meta, vision_msgs::msg::Detection2DArray &detection_array_msg)
{
    // Keeping this function mainly for ROS message population and general object handling.
    // The selected object identification logic is best kept in the osd_probe_callback itself
    // due to the need for immediate feedback to other parts of that probe.
    current_tracked_objects_[obj_meta->object_id] = obj_meta->rect_params;

    vision_msgs::msg::Detection2D detection;
    detection.header.stamp = detection_array_msg.header.stamp;
    detection.header.frame_id = detection_array_msg.header.frame_id;

    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = std::string(obj_meta->obj_label);
    hypothesis.hypothesis.score = obj_meta->confidence;
    detection.results.push_back(hypothesis);

    detection.bbox.center.position.x = obj_meta->rect_params.left + obj_meta->rect_params.width / 2.0;
    detection.bbox.center.position.y = obj_meta->rect_params.top + obj_meta->rect_params.height / 2.0;
    detection.bbox.size_x = obj_meta->rect_params.width;
    detection.bbox.size_y = obj_meta->rect_params.height;
    detection_array_msg.detections.push_back(detection);

    obj_meta->rect_params.border_width = 3;
    obj_meta->rect_params.has_bg_color = 0;
    obj_meta->rect_params.border_color = {0.0, 0.0, 1.0, 1.0};

    // This part is removed as selected object identification is handled in the probe.
    // return (obj_meta->object_id == selected_object_id_);
    return false; // Always return false if not identifying selected object here.
}
*/


void ObjectDetectionNode::manage_selected_object_kalman_filter(
    bool selected_object_found_in_frame,
    const NvOSD_RectParams &current_selected_bbox_detected,
    NvDsObjectMeta *current_selected_obj_meta_ptr,
    double &predicted_x, double &predicted_y,
    double &predicted_vx, double &predicted_vy)
{
    if (selected_object_id_ != UNTRACKED_OBJECT_ID) {
        // Retrieve DeepStream tracker's state for the selected object from user metadata
        selected_object_tracker_state_ = EMPTY; // Reset to EMPTY (lost) for current frame
        if (current_selected_obj_meta_ptr) {
            for (GList *l_user_meta = current_selected_obj_meta_ptr->obj_user_meta_list; l_user_meta != nullptr; l_user_meta = l_user_meta->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user_meta->data;
                if (user_meta->base_meta.meta_type == NVDS_TRACKER_METADATA) {
                    NvDsTargetMiscDataObject *selected_tracker_obj_data = (NvDsTargetMiscDataObject *)user_meta->user_meta_data;
                    if (selected_tracker_obj_data && selected_tracker_obj_data->numObj > 0) {
                        selected_object_tracker_state_ = selected_tracker_obj_data->list[0].trackerState;
                    }
                    break;
                }
            }
        }

        if (!selected_object_kf_initialized_ || !selected_object_kf_) {
            if (selected_object_found_in_frame) {
                selected_object_kf_ = std::make_unique<KalmanFilter2D>();
                double center_x = current_selected_bbox_detected.left + current_selected_bbox_detected.width / 2.0;
                double center_y = current_selected_bbox_detected.top + current_selected_bbox_detected.height / 2.0;
                selected_object_kf_->init(center_x, center_y);
                selected_object_kf_initialized_ = true;
                selected_object_lost_frames_ = 0;
                selected_object_last_bbox_ = current_selected_bbox_detected;
            } else {
                RCLCPP_DEBUG(this->get_logger(), "Selected object %lu not found (DS state: %d) and KF not initialized. Will deselect if not found soon.",
                             selected_object_id_, selected_object_tracker_state_);
            }
        } else {
            selected_object_kf_->predict();

            if (selected_object_found_in_frame) {
                double center_x = current_selected_bbox_detected.left + current_selected_bbox_detected.width / 2.0;
                double center_y = current_selected_bbox_detected.top + current_selected_bbox_detected.height / 2.0;
                selected_object_kf_->update(center_x, center_y);
                selected_object_lost_frames_ = 0;
                selected_object_last_bbox_ = current_selected_bbox_detected;
            } else {
                selected_object_lost_frames_++;
                RCLCPP_DEBUG(this->get_logger(), "Selected object ID %lu KF predicting. Frames lost: %u (DS state: %d)",
                             selected_object_id_, selected_object_lost_frames_, selected_object_tracker_state_);

                if (selected_object_lost_frames_ > KF_LOST_THRESHOLD && selected_object_tracker_state_ == EMPTY)
                {
                    RCLCPP_INFO(this->get_logger(), "Selected object ID %lu lost by KF (predicted for %u frames) and DeepStream tracker state is EMPTY. Deselecting.",
                                selected_object_id_, selected_object_lost_frames_);
                    selected_object_id_ = UNTRACKED_OBJECT_ID;
                    selected_object_kf_initialized_ = false;
                    selected_object_kf_.reset();
                    selected_object_lost_frames_ = 0;
                    selected_object_tracker_state_ = EMPTY;
                }
            }
        }

        if (selected_object_kf_initialized_ && selected_object_kf_) {
            predicted_x = selected_object_kf_->getX();
            predicted_y = selected_object_kf_->getY();
            predicted_vx = selected_object_kf_->getVx();
            predicted_vy = selected_object_kf_->getVy();

            selected_object_last_bbox_.left = predicted_x - selected_object_last_bbox_.width / 2.0;
            selected_object_last_bbox_.top = predicted_y - selected_object_last_bbox_.height / 2.0;
        } else {
            predicted_x = 0.0; predicted_y = 0.0; predicted_vx = 0.0; predicted_vy = 0.0;
        }
    }
    else {
        if (selected_object_kf_initialized_ || selected_object_kf_) {
            selected_object_kf_initialized_ = false;
            selected_object_kf_.reset();
            selected_object_lost_frames_ = 0;
            selected_object_tracker_state_ = EMPTY;
        }
        predicted_x = 0.0; predicted_y = 0.0; predicted_vx = 0.0; predicted_vy = 0.0;
    }
}

void ObjectDetectionNode::customize_object_osd(
    NvDsObjectMeta *obj_meta,
    bool is_selected,
    double predicted_vx, double predicted_vy,
    bool selected_object_found_in_frame)
{
    std::stringstream ss_label_conf;

    if (is_selected)
    {
        obj_meta->rect_params = selected_object_last_bbox_; // Use KF predicted bbox for drawing
        obj_meta->rect_params.border_color = {1.0, 0.0, 0.0, 1.0}; // Highlight selected object in Red

        ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id
                      << " Conf: " << std::fixed << std::setprecision(2) << obj_meta->confidence;

        ss_label_conf << " Vx: " << std::fixed << std::setprecision(2) << predicted_vx
                      << " Vy: " << std::fixed << std::setprecision(2) << predicted_vy;

        ss_label_conf << " (DS:" << [](TRACKER_STATE state) {
            switch(state) {
                case ACTIVE: return "Active";
                case INACTIVE: return "Inactive";
                case TENTATIVE: return "Tentative";
                case PROJECTED: return "Projected";
                case EMPTY: return "Lost";
                default: return "Unknown";
            }
        }(selected_object_tracker_state_) << ")";

        if (!selected_object_found_in_frame) {
            ss_label_conf << " (KF:Pred " << selected_object_lost_frames_ << " fr)";
        }
    } else {
        ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id
                      << " Conf: " << std::fixed << std::setprecision(2) << obj_meta->confidence;
        obj_meta->rect_params.border_color = {0.0, 0.0, 1.0, 1.0};
    }

    std::string label_conf_str = ss_label_conf.str();
    if (obj_meta->text_params.display_text)
    {
        g_free(obj_meta->text_params.display_text);
        obj_meta->text_params.display_text = nullptr;
    }
    obj_meta->text_params.display_text = g_strdup(label_conf_str.c_str());

    obj_meta->text_params.x_offset = (guint)obj_meta->rect_params.left;
    gint preferred_y_signed = (gint)(obj_meta->rect_params.top - 25);
    if (preferred_y_signed < 0) {
        obj_meta->text_params.y_offset = (guint)(obj_meta->rect_params.top + obj_meta->rect_params.height + 5);
    } else {
        obj_meta->text_params.y_offset = (guint)preferred_y_signed;
    }
    obj_meta->text_params.font_params.font_name = (gchar *)"Sans";
    obj_meta->text_params.font_params.font_size = 12;
    obj_meta->text_params.font_params.font_color = obj_meta->rect_params.border_color;
    obj_meta->text_params.set_bg_clr = 0;
}

void ObjectDetectionNode::draw_selected_object_overlay_if_occluded(
    NvDsBatchMeta *batch_meta,
    NvDsFrameMeta *frame_meta,
    double predicted_vx, double predicted_vy)
{
    NvOSD_RectParams &selected_rect_for_drawing = selected_object_last_bbox_;

    gfloat center_x = selected_rect_for_drawing.left + selected_rect_for_drawing.width / 2.0;
    gfloat center_y = selected_rect_for_drawing.top + selected_rect_for_drawing.height / 2.0;
    gfloat reticule_size = 20.0;

    NvDsDisplayMeta *reticule_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (reticule_display_meta)
    {
        reticule_display_meta->num_lines = 2;
        NvOSD_ColorParams reticule_color = {1.0, 0.0, 0.0, 1.0};

        reticule_display_meta->line_params[0].x1 = (guint)(center_x - reticule_size / 2.0);
        reticule_display_meta->line_params[0].y1 = (guint)center_y;
        reticule_display_meta->line_params[0].x2 = (guint)(center_x + reticule_size / 2.0);
        reticule_display_meta->line_params[0].y2 = (guint)center_y;
        reticule_display_meta->line_params[0].line_width = 2;
        reticule_display_meta->line_params[0].line_color = reticule_color;

        reticule_display_meta->line_params[1].x1 = (guint)center_x;
        reticule_display_meta->line_params[1].y1 = (guint)(center_y - reticule_size / 2.0);
        reticule_display_meta->line_params[1].x2 = (guint)center_x;
        reticule_display_meta->line_params[1].y2 = (guint)(center_y + reticule_size / 2.0);
        reticule_display_meta->line_params[1].line_width = 2;
        reticule_display_meta->line_params[1].line_color = reticule_color;

        nvds_add_display_meta_to_frame(frame_meta, reticule_display_meta);
    }

    NvDsDisplayMeta *predicted_bbox_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (predicted_bbox_display_meta) {
        predicted_bbox_display_meta->num_rects = 1;
        predicted_bbox_display_meta->rect_params[0] = selected_rect_for_drawing;
        predicted_bbox_display_meta->rect_params[0].border_color = {1.0, 0.0, 0.0, 1.0};
        predicted_bbox_display_meta->rect_params[0].border_width = 3;
        predicted_bbox_display_meta->rect_params[0].has_bg_color = 0;
        nvds_add_display_meta_to_frame(frame_meta, predicted_bbox_display_meta);
    }

    NvDsDisplayMeta *kf_text_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (kf_text_display_meta) {
        kf_text_display_meta->num_labels = 1;
        std::stringstream kf_text_ss;
        kf_text_ss << "ID: " << selected_object_id_
                   << " Vx: " << std::fixed << std::setprecision(2) << predicted_vx
                   << " Vy: " << std::fixed << std::setprecision(2) << predicted_vy
                   << " (DS:" << [](TRACKER_STATE state) {
                        switch(state) {
                            case ACTIVE: return "Active";
                            case INACTIVE: return "Inactive";
                            case TENTATIVE: return "Tentative";
                            case PROJECTED: return "Projected";
                            case EMPTY: return "Lost";
                            default: return "Unknown";
                        }
                    }(selected_object_tracker_state_) << ")"
                   << " (KF:Pred " << selected_object_lost_frames_ << " fr)";

        kf_text_display_meta->text_params[0].display_text = g_strdup(kf_text_ss.str().c_str());
        kf_text_display_meta->text_params[0].x_offset = (guint)(selected_rect_for_drawing.left);
        gint text_y_offset_signed = (gint)(selected_rect_for_drawing.top - 25);
        if (text_y_offset_signed < 0) {
            kf_text_display_meta->text_params[0].y_offset = (guint)(selected_rect_for_drawing.top + selected_rect_for_drawing.height + 5);
        } else {
            kf_text_display_meta->text_params[0].y_offset = (guint)text_y_offset_signed;
        }
        kf_text_display_meta->text_params[0].font_params.font_name = (gchar *)"Sans";
        kf_text_display_meta->text_params[0].font_params.font_size = 12;
        kf_text_display_meta->text_params[0].font_params.font_color = {1.0, 1.0, 1.0, 1.0};
        kf_text_display_meta->text_params[0].set_bg_clr = 0;
        nvds_add_display_meta_to_frame(frame_meta, kf_text_display_meta);
    }
}

// --- Method to cycle through detected targets ---
void ObjectDetectionNode::cycle_selected_target(bool forward)
{
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);

    if (current_tracked_objects_.empty())
    {
        selected_object_id_ = UNTRACKED_OBJECT_ID;
        selected_object_kf_initialized_ = false;
        selected_object_kf_.reset();
        RCLCPP_INFO(this->get_logger(), "No objects currently detected to select.");
        return;
    }

    std::vector<guint64> object_ids;
    for (const auto& pair : current_tracked_objects_)
    {
        object_ids.push_back(pair.first);
    }
    std::sort(object_ids.begin(), object_ids.end());

    auto it = std::find(object_ids.begin(), object_ids.end(), selected_object_id_);

    if (it == object_ids.end() || selected_object_id_ == UNTRACKED_OBJECT_ID)
    {
        if (forward) {
            selected_object_id_ = object_ids[0];
        } else {
            selected_object_id_ = object_ids.back();
        }
    }
    else
    {
        if (forward) {
            ++it;
            if (it == object_ids.end())
            {
                selected_object_id_ = object_ids[0];
            }
            else
            {
                selected_object_id_ = *it;
            }
        } else {
            if (it == object_ids.begin())
            {
                selected_object_id_ = object_ids.back();
            }
            else
            {
                --it;
                selected_object_id_ = *it;
            }
        }
    }
    RCLCPP_INFO(this->get_logger(), "New selected object ID: %lu", selected_object_id_);
    selected_object_kf_initialized_ = false;
    selected_object_kf_.reset();
    selected_object_lost_frames_ = 0;
    selected_object_tracker_state_ = EMPTY;
}

// --- Callback for joystick input ---
void ObjectDetectionNode::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    const int FORWARD_BUTTON_INDEX = 0;
    const int BACKWARD_BUTTON_INDEX = 1;

    bool current_button0_pressed = (msg->buttons.size() > FORWARD_BUTTON_INDEX && msg->buttons[FORWARD_BUTTON_INDEX] == 1);
    bool current_button1_pressed = (msg->buttons.size() > BACKWARD_BUTTON_INDEX && msg->buttons[BACKWARD_BUTTON_INDEX] == 1);

    if (current_button0_pressed && !button0_pressed_prev_)
    {
        RCLCPP_INFO(this->get_logger(), "Joystick button %d pressed. Cycling target forward.", FORWARD_BUTTON_INDEX);
        cycle_selected_target(true);
    }
    else if (current_button1_pressed && !button1_pressed_prev_)
    {
        RCLCPP_INFO(this->get_logger(), "Joystick button %d pressed. Cycling target backward.", BACKWARD_BUTTON_INDEX);
        cycle_selected_target(false);
    }

    button0_pressed_prev_ = current_button0_pressed;
    button1_pressed_prev_ = current_button1_pressed;
}

// --- Main entrypoint ---
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;

    std::shared_ptr<ObjectDetectionNode> node = nullptr;
    try {
        node = std::make_shared<ObjectDetectionNode>(options);
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Node initialization or runtime error: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "An unknown error occurred during node execution.");
    }

    rclcpp::shutdown();
    return 0;
}