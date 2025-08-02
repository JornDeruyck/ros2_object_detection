// src/object_detection.cpp
#include "ros2_object_detection/object_detection.hpp"
#include "ros2_object_detection/osd_renderer.hpp"
#include "ros2_object_detection/constants.hpp"

#include <algorithm>
#include <set>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <cmath>
#include <cstring>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <glib.h>

#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "nvll_osd_struct.h"
#include "nvds_tracker_meta.h"

#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

// Helper function to convert NvBbox_Coords to NvOSD_RectParams
static NvOSD_RectParams bbox_coords_to_rect_params(const NvBbox_Coords& coords) {
    NvOSD_RectParams params;
    params.left = coords.left;
    params.top = coords.top;
    params.width = coords.width;
    params.height = coords.height;
    params.border_width = 0;
    params.has_bg_color = 0;
    params.bg_color = {0,0,0,0};
    return params;
}

GstFlowReturn ObjectDetectionNode::new_sample_callback(GstElement *sink, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstSample *sample = nullptr;
    g_signal_emit_by_name(sink, "pull-sample", &sample);
    if (!sample) return GST_FLOW_OK;
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        auto msg = sensor_msgs::msg::CompressedImage();
        msg.header.stamp = node->get_clock()->now();
        msg.header.frame_id = "camera_frame";
        msg.format = "jpeg";
        msg.data.assign(map.data, map.data + map.size);
        node->compressed_publisher_->publish(msg);
        gst_buffer_unmap(buffer, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

GstPadProbeReturn ObjectDetectionNode::osd_probe_callback(GstPad * /*pad*/, GstPadProbeInfo *info, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstBuffer *gst_buffer = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header.stamp = node->get_clock()->now();
    detection_array_msg.header.frame_id = "camera_frame";

    for (GList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) continue;

        if (node->osd_renderer_) node->osd_renderer_->update_and_display_fps(batch_meta, frame_meta);

        std::lock_guard<std::mutex> objects_lock(node->tracked_objects_mutex_);
        node->current_tracked_objects_.clear();
        node->current_tracked_classes_.clear();
        
        const NvDsObjectMeta* selected_obj_meta = nullptr;

        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue;

            bool is_allowed_class = node->allowed_class_ids_.empty();
            if (!is_allowed_class) {
                for (long int allowed_id : node->allowed_class_ids_) {
                    if (obj_meta->class_id == allowed_id) {
                        is_allowed_class = true;
                        break;
                    }
                }
            }

            if (!is_allowed_class) {
                obj_meta->rect_params.border_width = 0;
                if (obj_meta->text_params.display_text) {
                    g_free(obj_meta->text_params.display_text);
                    obj_meta->text_params.display_text = nullptr;
                }
                continue;
            }

            node->current_tracked_objects_[obj_meta->object_id] = bbox_coords_to_rect_params(obj_meta->tracker_bbox_info.org_bbox_coords);
            node->current_tracked_classes_[obj_meta->object_id] = std::string(obj_meta->obj_label);
            node->populate_ros_detection_message(obj_meta, detection_array_msg);

            if (obj_meta->object_id == node->selected_object_id_) {
                selected_obj_meta = obj_meta;
                if (obj_meta->text_params.display_text) {
                    g_free(obj_meta->text_params.display_text);
                    obj_meta->text_params.display_text = nullptr;
                }
                obj_meta->rect_params.border_width = 0;
            } else {
                if (node->osd_renderer_) {
                    node->osd_renderer_->render_non_selected_object_osd(batch_meta, frame_meta, obj_meta);
                }
            }
        }

        if (node->selected_object_id_ != NO_OBJECT_ID) {
            OSDTrackingStatus status = node->manage_selected_object_state(selected_obj_meta);
            
            bool is_locked = (node->selected_object_id_ == node->locked_target_id_);
            double pred_vx = 0.0, pred_vy = 0.0;
            NvOSD_RectParams bbox_to_render = {};

            if (node->selected_object_kf_initialized_) {
                pred_vx = node->selected_object_kf_->getVx();
                pred_vy = node->selected_object_kf_->getVy();
                bbox_to_render = node->selected_object_last_bbox_;
                bbox_to_render.left = node->selected_object_kf_->getX() - bbox_to_render.width / 2.0;
                bbox_to_render.top = node->selected_object_kf_->getY() - bbox_to_render.height / 2.0;
            }
            
            if (node->osd_renderer_ && node->selected_object_kf_initialized_) {
                node->osd_renderer_->render_selected_object_osd(
                    batch_meta, frame_meta, node->selected_object_id_, node->selected_object_class_label_,
                    status, is_locked, bbox_to_render, node->selected_object_lost_frames_,
                    pred_vx, pred_vy, node->camera_fov_rad_
                );
            }
        }
    }

    if (!detection_array_msg.detections.empty()) {
        node->detection_publisher_->publish(detection_array_msg);
    }

    return GST_PAD_PROBE_OK;
}

ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions &options)
    : Node("object_detection_node", options),
      pipeline_(nullptr), main_loop_(nullptr),
      selected_object_id_(NO_OBJECT_ID), locked_target_id_(NO_OBJECT_ID),
      selected_object_kf_(nullptr), selected_object_kf_initialized_(false),
      selected_object_lost_frames_(0),
      button0_pressed_prev_(false), button1_pressed_prev_(false), button2_pressed_prev_(false)
{
    RCLCPP_INFO(this->get_logger(), "Initializing ObjectDetectionNode...");
    memset(&selected_object_last_bbox_, 0, sizeof(NvOSD_RectParams));
    this->declare_parameter<std::string>("pipeline_string", "");
    this->declare_parameter<std::vector<long int>>("allowed_class_ids", std::vector<long int>());
    this->declare_parameter<std::string>("detection_topic", "detections");
    this->declare_parameter<std::string>("selected_target_topic", "selected_target_id");
    this->declare_parameter<std::string>("image_topic", "image_raw/compressed");
    this->declare_parameter<std::string>("joy_topic", "/joy");
    this->declare_parameter<bool>("use_qos_reliable", true);
    this->declare_parameter<int>("qos_history_depth", 1);
    this->declare_parameter<double>("camera_fov", 90.0);

    std::string pipeline_string = this->get_parameter("pipeline_string").as_string();
    allowed_class_ids_ = this->get_parameter("allowed_class_ids").as_integer_array();
    std::string detection_topic = this->get_parameter("detection_topic").as_string();
    std::string selected_target_topic = this->get_parameter("selected_target_topic").as_string();
    std::string image_topic = this->get_parameter("image_topic").as_string();
    std::string joy_topic = this->get_parameter("joy_topic").as_string();
    bool use_qos_reliable = this->get_parameter("use_qos_reliable").as_bool();
    int qos_history_depth = this->get_parameter("qos_history_depth").as_int();
    double camera_fov_deg = this->get_parameter("camera_fov").as_double();
    camera_fov_rad_ = camera_fov_deg * M_PI / 180.0;

    if (pipeline_string.empty()) {
        RCLCPP_FATAL(this->get_logger(), "Parameter 'pipeline_string' is empty.");
        throw std::runtime_error("Empty 'pipeline_string' parameter.");
    }

    rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(qos_history_depth));
    if (use_qos_reliable) qos_profile.reliable(); else qos_profile.best_effort();
    qos_profile.durability_volatile();

    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(detection_topic, qos_profile);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(image_topic, qos_profile);
    selected_target_publisher_ = this->create_publisher<std_msgs::msg::UInt64>(selected_target_topic, qos_profile);
    joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(joy_topic, 10, std::bind(&ObjectDetectionNode::joy_callback, this, std::placeholders::_1));
    osd_renderer_ = std::make_unique<OSDRenderer>(this);
    gst_init(nullptr, nullptr);
    main_loop_ = g_main_loop_new(nullptr, FALSE);
    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_string.c_str(), &error);
    if (!pipeline_) {
        RCLCPP_FATAL(this->get_logger(), "Failed to parse GStreamer pipeline: %s", error ? error->message : "Unknown error");
        if (error) g_error_free(error);
        throw std::runtime_error("GStreamer pipeline parsing failed.");
    }
    GstElement *osd_element = gst_bin_get_by_name(GST_BIN(pipeline_), "nvdsosd0");
    if (osd_element) {
        GstPad *osd_sink_pad = gst_element_get_static_pad(osd_element, "sink");
        if (osd_sink_pad) {
            gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_probe_callback, this, nullptr);
            gst_object_unref(osd_sink_pad);
        }
        gst_object_unref(osd_element);
    }
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline_), "ros_sink");
    if (appsink) {
        g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample_callback), this);
        gst_object_unref(appsink);
    }
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    gstreamer_thread_ = std::thread([this]() { g_main_loop_run(main_loop_); });
    RCLCPP_INFO(this->get_logger(), "ObjectDetectionNode fully initialized.");
}

ObjectDetectionNode::~ObjectDetectionNode()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down ObjectDetectionNode...");
    if (main_loop_ && g_main_loop_is_running(main_loop_)) g_main_loop_quit(main_loop_);
    if (gstreamer_thread_.joinable()) gstreamer_thread_.join();
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }
    if (main_loop_) g_main_loop_unref(main_loop_);
    RCLCPP_INFO(this->get_logger(), "ObjectDetectionNode shut down complete.");
}

void ObjectDetectionNode::populate_ros_detection_message(NvDsObjectMeta *obj_meta, vision_msgs::msg::Detection2DArray &detection_array_msg)
{
    vision_msgs::msg::Detection2D detection;
    detection.header.stamp = detection_array_msg.header.stamp;
    detection.header.frame_id = detection_array_msg.header.frame_id;
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = std::string(obj_meta->obj_label);
    hypothesis.hypothesis.score = obj_meta->confidence;
    detection.results.push_back(hypothesis);
    const auto& tracker_bbox = obj_meta->tracker_bbox_info.org_bbox_coords;
    detection.bbox.center.position.x = tracker_bbox.left + tracker_bbox.width / 2.0;
    detection.bbox.center.position.y = tracker_bbox.top + tracker_bbox.height / 2.0;
    detection.bbox.size_x = tracker_bbox.width;
    detection.bbox.size_y = tracker_bbox.height;
    detection.id = std::to_string(obj_meta->object_id);
    detection_array_msg.detections.push_back(detection);
}

void ObjectDetectionNode::cycle_selected_target(bool forward)
{
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);

    // If there are no objects, always deselect.
    if (current_tracked_objects_.empty()) {
        if (selected_object_id_ != NO_OBJECT_ID) {
            RCLCPP_INFO(this->get_logger(), "No objects detected. Deselecting target.");
            selected_object_id_ = NO_OBJECT_ID;
        }
        // Reset KF state when deselecting
        selected_object_kf_initialized_ = false;
        selected_object_kf_.reset();
        selected_object_lost_frames_ = 0;
        return;
    }

    std::vector<guint64> object_ids;
    for (const auto& pair : current_tracked_objects_) {
        object_ids.push_back(pair.first);
    }
    std::sort(object_ids.begin(), object_ids.end());

    auto it = std::find(object_ids.begin(), object_ids.end(), selected_object_id_);

    // Case 1: No object is currently selected.
    if (it == object_ids.end()) {
        selected_object_id_ = forward ? object_ids.front() : object_ids.back();
        RCLCPP_INFO(this->get_logger(), "No object selected. Selecting first/last: %lu", selected_object_id_);
    }
    // Case 2: An object is selected.
    else {
        if (forward) {
            it++; // Move to next position
            // If we were at the last element, deselect.
            if (it == object_ids.end()) {
                RCLCPP_INFO(this->get_logger(), "Cycled past last object. Deselecting.");
                selected_object_id_ = NO_OBJECT_ID;
            } else {
                selected_object_id_ = *it;
                RCLCPP_INFO(this->get_logger(), "Cycled forward to new object: %lu", selected_object_id_);
            }
        } else { // backward
            // If we were at the first element, deselect.
            if (it == object_ids.begin()) {
                RCLCPP_INFO(this->get_logger(), "Cycled before first object. Deselecting.");
                selected_object_id_ = NO_OBJECT_ID;
            } else {
                it--; // Move to previous
                selected_object_id_ = *it;
                RCLCPP_INFO(this->get_logger(), "Cycled backward to new object: %lu", selected_object_id_);
            }
        }
    }

    // Reset KF state on any change of selection (including deselection).
    selected_object_kf_initialized_ = false;
    selected_object_kf_.reset();
    selected_object_lost_frames_ = 0;
}

void ObjectDetectionNode::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    const int FORWARD_BUTTON_INDEX = 0;
    const int BACKWARD_BUTTON_INDEX = 1;
    const int LOCK_BUTTON_INDEX = 2;
    bool fwd_pressed = (msg->buttons.size() > FORWARD_BUTTON_INDEX && msg->buttons[FORWARD_BUTTON_INDEX] == 1);
    bool back_pressed = (msg->buttons.size() > BACKWARD_BUTTON_INDEX && msg->buttons[BACKWARD_BUTTON_INDEX] == 1);
    bool lock_pressed = (msg->buttons.size() > LOCK_BUTTON_INDEX && msg->buttons[LOCK_BUTTON_INDEX] == 1);
    if (fwd_pressed && !button0_pressed_prev_) cycle_selected_target(true);
    if (back_pressed && !button1_pressed_prev_) cycle_selected_target(false);
    if (lock_pressed && !button2_pressed_prev_) {
        if (locked_target_id_ == selected_object_id_ && locked_target_id_ != NO_OBJECT_ID) {
            locked_target_id_ = NO_OBJECT_ID;
            RCLCPP_INFO(this->get_logger(), "Joystick: Unlocking target.");
        } else if (selected_object_id_ != NO_OBJECT_ID) {
            locked_target_id_ = selected_object_id_;
            RCLCPP_INFO(this->get_logger(), "Joystick: Locking target ID %lu.", locked_target_id_);
        } else {
            locked_target_id_ = NO_OBJECT_ID;
            RCLCPP_WARN(this->get_logger(), "Joystick: No target selected to lock.");
        }
        auto id_msg = std_msgs::msg::UInt64();
        id_msg.data = locked_target_id_;
        selected_target_publisher_->publish(id_msg);
    }
    button0_pressed_prev_ = fwd_pressed;
    button1_pressed_prev_ = back_pressed;
    button2_pressed_prev_ = lock_pressed;
}

OSDTrackingStatus ObjectDetectionNode::manage_selected_object_state(const NvDsObjectMeta* selected_obj_meta)
{
    if (!selected_object_kf_initialized_) {
        if (selected_obj_meta) {
            const auto& tracker_bbox = selected_obj_meta->tracker_bbox_info.org_bbox_coords;
            double center_x = tracker_bbox.left + tracker_bbox.width / 2.0;
            double center_y = tracker_bbox.top + tracker_bbox.height / 2.0;
            selected_object_kf_ = std::make_unique<KalmanFilter2D>();
            selected_object_kf_->init(center_x, center_y);
            selected_object_kf_initialized_ = true;
            selected_object_class_label_ = std::string(selected_obj_meta->obj_label);
            selected_object_last_bbox_ = bbox_coords_to_rect_params(tracker_bbox);
        } else {
            return OSDTrackingStatus::DETECTED;
        }
    }

    selected_object_kf_->predict();

    if (selected_obj_meta) {
        const auto& tracker_bbox = selected_obj_meta->tracker_bbox_info.org_bbox_coords;
        double center_x = tracker_bbox.left + tracker_bbox.width / 2.0;
        double center_y = tracker_bbox.top + tracker_bbox.height / 2.0;
        selected_object_kf_->update(center_x, center_y);
        
        selected_object_lost_frames_ = 0;
        selected_object_last_bbox_ = bbox_coords_to_rect_params(tracker_bbox);
        selected_object_class_label_ = std::string(selected_obj_meta->obj_label);
        return OSDTrackingStatus::DETECTED;
    } else {
        selected_object_lost_frames_++;
        if (selected_object_lost_frames_ > KF_LOST_THRESHOLD) {
            RCLCPP_INFO(this->get_logger(), "Target %lu lost for too long. Deselecting.", selected_object_id_);
            if (selected_object_id_ == locked_target_id_) {
                locked_target_id_ = NO_OBJECT_ID;
                auto msg = std_msgs::msg::UInt64();
                msg.data = 0;
                selected_target_publisher_->publish(msg);
            }
            selected_object_id_ = NO_OBJECT_ID;
            selected_object_kf_initialized_ = false;
            selected_object_kf_.reset();
            return OSDTrackingStatus::DETECTED;
        }
        return OSDTrackingStatus::OCCLUDED;
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    try {
        auto node = std::make_shared<ObjectDetectionNode>(options);
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Node error: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
