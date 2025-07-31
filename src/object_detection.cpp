// src/object_detection.cpp
#include "ros2_object_detection/object_detection.hpp"
#include "ros2_object_detection/kalman_filter_2d.hpp"
#include "ros2_object_detection/constants.hpp"
#include "ros2_object_detection/osd_renderer.hpp"

// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <cmath>
#include <cstring>
#include <vector>

// GStreamer & GLib includes
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <glib.h>

// DeepStream metadata headers
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "nvll_osd_struct.h"
#include "nvds_tracker_meta.h"

// ROS 2 message types
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

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

        // Use OSD Renderer for FPS display
        if (node->osd_renderer_) {
            node->osd_renderer_->update_and_display_fps(batch_meta, frame_meta);
        }

        std::lock_guard<std::mutex> objects_lock(node->tracked_objects_mutex_);
        node->current_tracked_objects_.clear(); // Clear detected objects from previous frame

        bool selected_object_found_in_frame = false;
        NvOSD_RectParams current_selected_bbox_detected = {}; // Initialize with default values
        NvDsObjectMeta *current_selected_obj_meta_ptr = nullptr;

        // First pass: Process all detected objects, filter by class, populate ROS message, and identify selected one if present
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue;

            // --- Class Filtering Logic ---
            bool is_allowed_class = false;
            if (node->allowed_class_ids_.empty()) {
                // If allowed_class_ids_ is empty, allow all classes
                is_allowed_class = true;
            } else {
                // Check if the object's class ID is in the allowed list
                for (int allowed_id : node->allowed_class_ids_) {
                    if (obj_meta->class_id == allowed_id) {
                        is_allowed_class = true;
                        break;
                    }
                }
            }

            if (!is_allowed_class) {
                // If this object's class is not allowed, skip it entirely.
                obj_meta->rect_params.border_width = 0;
                obj_meta->rect_params.has_bg_color = 0;
                if (obj_meta->text_params.display_text) {
                    g_free(obj_meta->text_params.display_text);
                    obj_meta->text_params.display_text = nullptr;
                }
                continue;
            }
            // --- END Class Filtering Logic ---

            node->current_tracked_objects_[obj_meta->object_id] = obj_meta->rect_params;
            node->populate_ros_detection_message(obj_meta, detection_array_msg);

            // Identify the selected object from the current frame's detections
            if (obj_meta->object_id == node->selected_object_id_) {
                selected_object_found_in_frame = true;
                current_selected_bbox_detected = obj_meta->rect_params;
                current_selected_obj_meta_ptr = obj_meta;

                // For the selected object, we will explicitly manage its OSD later.
                if (obj_meta->text_params.display_text) {
                    g_free(obj_meta->text_params.display_text);
                    obj_meta->text_params.display_text = nullptr;
                }
                obj_meta->rect_params.border_width = 0;
            } else {
                // For non-selected objects that passed the class filter, apply standard OSD
                if (node->osd_renderer_) {
                    node->osd_renderer_->render_non_selected_object_osd(obj_meta);
                }
            }
        }

        double predicted_x = 0.0, predicted_y = 0.0;
        double predicted_vx = 0.0, predicted_vy = 0.0;

        // Manage Kalman Filter state for the selected object (prediction/update/deselection)
        node->manage_selected_object_kalman_filter(
            selected_object_found_in_frame,
            current_selected_bbox_detected,
            current_selected_obj_meta_ptr,
            predicted_x, predicted_y, predicted_vx, predicted_vy);

        // After processing all objects and updating KF, render OSD for the selected object if one is active.
        if (node->selected_object_id_ != UNTRACKED_OBJECT_ID && node->osd_renderer_) {
            node->osd_renderer_->render_selected_object_osd(
                batch_meta,
                frame_meta,
                node->selected_object_id_,
                node->selected_object_kf_initialized_,
                node->selected_object_last_bbox_,
                node->selected_object_lost_frames_,
                node->selected_object_tracker_state_,
                predicted_vx,
                predicted_vy,
                selected_object_found_in_frame,
                current_selected_obj_meta_ptr
            );
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
      pipeline_(nullptr),
      main_loop_(nullptr),
      selected_object_id_(UNTRACKED_OBJECT_ID),
      selected_object_kf_(nullptr),
      selected_object_kf_initialized_(false),
      selected_object_lost_frames_(0),
      selected_object_tracker_state_(EMPTY),
      button0_pressed_prev_(false),
      button1_pressed_prev_(false)
{
    RCLCPP_INFO(this->get_logger(), "Initializing ObjectDetectionNode...");

    memset(&selected_object_last_bbox_, 0, sizeof(NvOSD_RectParams));

    // 1. Declare ALL Parameters
    this->declare_parameter<std::string>("pipeline_string", "");
    this->declare_parameter<std::vector<int>>("allowed_class_ids", std::vector<int>());
    this->declare_parameter<std::string>("detection_topic", "detections");
    this->declare_parameter<std::string>("image_topic", "image_raw/compressed");
    this->declare_parameter<std::string>("joy_topic", "/joy");
    this->declare_parameter<bool>("use_qos_reliable", true);
    this->declare_parameter<int>("qos_history_depth", 1);


    // 2. Retrieve Parameter Values
    std::string pipeline_string = this->get_parameter("pipeline_string").as_string();
    allowed_class_ids_ = this->get_parameter("allowed_class_ids").as_integer_array();
    std::string detection_topic = this->get_parameter("detection_topic").as_string();
    std::string image_topic = this->get_parameter("image_topic").as_string();
    std::string joy_topic = this->get_parameter("joy_topic").as_string();
    bool use_qos_reliable = this->get_parameter("use_qos_reliable").as_bool();
    int qos_history_depth = this->get_parameter("qos_history_depth").as_int();

    // Log the loaded parameters
    RCLCPP_INFO(this->get_logger(), "Loaded Parameters:");
    RCLCPP_INFO(this->get_logger(), "  pipeline_string: %s", pipeline_string.c_str());
    RCLCPP_INFO(this->get_logger(), "  detection_topic: %s", detection_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  image_topic: %s", image_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  joy_topic: %s", joy_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  use_qos_reliable: %s", use_qos_reliable ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  qos_history_depth: %d", qos_history_depth);

    if (allowed_class_ids_.empty()) {
        RCLCPP_INFO(this->get_logger(), "  allowed_class_ids: (empty - all classes allowed)");
    } else {
        std::stringstream ss;
        ss << "  allowed_class_ids: [";
        for (size_t i = 0; i < allowed_class_ids_.size(); ++i) {
            ss << allowed_class_ids_[i];
            if (i < allowed_class_ids_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }

    if (pipeline_string.empty()) {
        RCLCPP_FATAL(this->get_logger(), "Parameter 'pipeline_string' is empty. Please provide a GStreamer pipeline string.");
        throw std::runtime_error("Empty 'pipeline_string' parameter.");
    }

    // Configure QoS profile based on parameters
    rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(qos_history_depth));
    if (use_qos_reliable) {
        qos_profile.reliable();
    } else {
        qos_profile.best_effort();
    }
    qos_profile.durability_volatile();

    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(detection_topic, qos_profile);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(image_topic, qos_profile);

    joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
        joy_topic, 10, std::bind(&ObjectDetectionNode::joy_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribing to %s topic for joystick input.", joy_topic.c_str());

    // Initialize OSDRenderer, passing a raw pointer to this node
    // This avoids the bad_weak_ptr issue by not calling shared_from_this() in the constructor.
    osd_renderer_ = std::make_unique<OSDRenderer>(this); // Changed to pass 'this' raw pointer

    gst_init(nullptr, nullptr);
    main_loop_ = g_main_loop_new(nullptr, FALSE);

    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_string.c_str(), &error);
    if (!pipeline_)
    {
        RCLCPP_FATAL(this->get_logger(), "Failed to parse GStreamer pipeline: %s", error ? error->message : "Unknown error");
        if (error) g_error_free(error);
        throw std::runtime_error("GStreamer pipeline parsing failed.");
    }

    GstElement *osd_element = gst_bin_get_by_name(GST_BIN(pipeline_), "nvdsosd0");
    if (osd_element)
    {
        GstPad *osd_sink_pad = gst_element_get_static_pad(osd_element, "sink");
        if (osd_sink_pad) {
            gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_probe_callback, this, nullptr);
            gst_object_unref(osd_sink_pad);
            RCLCPP_INFO(this->get_logger(), "Attached OSD probe to 'nvdsosd0' sink pad.");
        } else {
            RCLCPP_WARN(this->get_logger(), "Could not get sink pad from 'nvdsosd0' element for probe attachment. Custom OSD will not function.");
        }
        gst_object_unref(osd_element);
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find 'nvdsosd0' element in pipeline. Custom OSD will not be rendered.");
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

    RCLCPP_INFO(this->get_logger(), "ObjectDetectionNode fully initialized.");
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

// --- Private Methods ---

void ObjectDetectionNode::populate_ros_detection_message(NvDsObjectMeta *obj_meta, vision_msgs::msg::Detection2DArray &detection_array_msg)
{
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
    detection.id = std::to_string(obj_meta->object_id);
    detection_array_msg.detections.push_back(detection);
}

void ObjectDetectionNode::manage_selected_object_kalman_filter(
    bool selected_object_found_in_frame,
    const NvOSD_RectParams &current_selected_bbox_detected,
    NvDsObjectMeta *current_selected_obj_meta_ptr,
    double &predicted_x, double &predicted_y,
    double &predicted_vx, double &predicted_vy)
{
    if (selected_object_id_ != UNTRACKED_OBJECT_ID) {
        if (selected_object_found_in_frame && current_selected_obj_meta_ptr) {
            bool tracker_meta_found = false;
            RCLCPP_DEBUG(this->get_logger(), "Selected Object ID %lu: %s. Iterating through user metadata.",
                         selected_object_id_,
                         current_selected_obj_meta_ptr->obj_user_meta_list == nullptr ? "obj_user_meta_list is NULL" : "obj_user_meta_list is NOT NULL");

            for (GList *l_user_meta = current_selected_obj_meta_ptr->obj_user_meta_list; l_user_meta != nullptr; l_user_meta = l_user_meta->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user_meta->data;
                RCLCPP_DEBUG(this->get_logger(), "  Found user meta type: %d", user_meta->base_meta.meta_type);
                if (user_meta->base_meta.meta_type == NVDS_TRACKER_METADATA) {
                    NvDsTargetMiscDataObject *selected_tracker_obj_data = (NvDsTargetMiscDataObject *)user_meta->user_meta_data;
                    if (selected_tracker_obj_data && selected_tracker_obj_data->numObj > 0) {
                        selected_object_tracker_state_ = selected_tracker_obj_data->list[0].trackerState;
                        tracker_meta_found = true;
                    }
                    break;
                }
            }
            if (!tracker_meta_found) {
                RCLCPP_DEBUG(this->get_logger(), "Selected object ID %lu does not have state meta data in this frame. Setting tracker state to EMPTY.",
                             selected_object_id_);
                selected_object_tracker_state_ = EMPTY;
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
                RCLCPP_DEBUG(this->get_logger(), "Selected object %lu not found and KF not initialized. Will deselect if not found soon.",
                             selected_object_id_);
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
        }
        selected_object_tracker_state_ = EMPTY;
        predicted_x = 0.0; predicted_y = 0.0; predicted_vx = 0.0; predicted_vy = 0.0;
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
        selected_object_lost_frames_ = 0;
        selected_object_tracker_state_ = EMPTY;
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
