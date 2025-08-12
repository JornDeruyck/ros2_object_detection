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
#include "std_msgs/msg/u_int64.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include "std_srvs/srv/trigger.hpp"

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

// --- Class Constructor and Destructor (Restructured) ---

ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions &options)
    : Node("object_detection_node", options),
      pipeline_(nullptr), main_loop_(nullptr),
      selected_object_id_(NO_OBJECT_ID), locked_target_id_(NO_OBJECT_ID)
{
    RCLCPP_INFO(this->get_logger(), "Initializing ObjectDetectionNode...");
    this->declare_parameter<std::string>("pipeline_string", "");
    this->declare_parameter<std::vector<long int>>("allowed_class_ids", std::vector<long int>());
    this->declare_parameter<std::string>("image_topic", "image_raw/compressed");
    this->declare_parameter<bool>("use_qos_reliable", true);
    this->declare_parameter<int>("qos_history_depth", 1);
    this->declare_parameter<double>("camera_fov", 90.0);

    std::string pipeline_string = this->get_parameter("pipeline_string").as_string();
    allowed_class_ids_ = this->get_parameter("allowed_class_ids").as_integer_array();
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

    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", qos_profile);
    target_publisher_ = this->create_publisher<vision_msgs::msg::Detection2D>("~/target", qos_profile);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("~/image_compressed", qos_profile);
    osd_renderer_ = std::make_unique<OSDRenderer>(this);

    lock_target_service_ = this->create_service<std_srvs::srv::Trigger>(
        "~/lock_target",
        std::bind(&ObjectDetectionNode::handle_lock_target, this, std::placeholders::_1, std::placeholders::_2));
    
    unlock_target_service_ = this->create_service<std_srvs::srv::Trigger>(
        "~/unlock_target",
        std::bind(&ObjectDetectionNode::handle_unlock_target, this, std::placeholders::_1, std::placeholders::_2));

    cycle_target_forward_service_ = this->create_service<std_srvs::srv::Trigger>(
        "~/cycle_target_forward",
        std::bind(&ObjectDetectionNode::handle_cycle_forward, this, std::placeholders::_1, std::placeholders::_2));
    
    cycle_target_backward_service_ = this->create_service<std_srvs::srv::Trigger>(
        "~/cycle_target_backward",
        std::bind(&ObjectDetectionNode::handle_cycle_backward, this, std::placeholders::_1, std::placeholders::_2));

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

// --- GStreamer Callbacks ---

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

    const rclcpp::Time current_stamp = node->get_clock()->now();
    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header.stamp = current_stamp;
    detection_array_msg.header.frame_id = "camera_frame";

    std::lock_guard<std::mutex> lock(node->tracked_objects_mutex_);

    // Step 1: Mark all existing objects as potentially unseen and create a map of current frame metadata
    std::map<guint64, NvDsObjectMeta*> current_frame_meta_map;
    for (auto& pair : node->persistent_object_map_) {
        pair.second.frames_since_seen++;
    }

    // Step 2: Update map with currently visible objects
    for (GList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) continue;

        if (node->osd_renderer_) {
            double center_x = frame_meta->source_frame_width / 2.0;
            double center_y = frame_meta->source_frame_height / 2.0;
            double crosshair_size = 50.0;
            node->osd_renderer_->update_and_display_fps(batch_meta, frame_meta);
            node->osd_renderer_->draw_reticule(batch_meta, frame_meta, center_x, center_y, crosshair_size, node->osd_renderer_->white_color_, 2, ReticuleStyle::CROSS_GAP);
        }

        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue;

            // Clear default OSD first
            obj_meta->rect_params.border_width = 0;
            obj_meta->rect_params.has_bg_color = 0;
            if (obj_meta->text_params.display_text) {
                g_free(obj_meta->text_params.display_text);
                obj_meta->text_params.display_text = nullptr;
            }

            bool is_allowed_class = node->allowed_class_ids_.empty() ||
                (std::find(node->allowed_class_ids_.begin(), node->allowed_class_ids_.end(), obj_meta->class_id) != node->allowed_class_ids_.end());
            
            if (is_allowed_class) {
                // Update state in persistent map
                auto& object_state = node->persistent_object_map_[obj_meta->object_id];
                object_state.id = obj_meta->object_id;
                object_state.class_label = std::string(obj_meta->obj_label);
                object_state.confidence = obj_meta->confidence;
                object_state.last_bbox = bbox_coords_to_rect_params(obj_meta->tracker_bbox_info.org_bbox_coords);
                object_state.frames_since_seen = 0;

                double center_x = object_state.last_bbox.left + object_state.last_bbox.width / 2.0;
                double center_y = object_state.last_bbox.top + object_state.last_bbox.height / 2.0;
                if (!object_state.kf_initialized) {
                    object_state.kf = std::make_unique<KalmanFilter2D>();
                    object_state.kf->init(center_x, center_y);
                    object_state.kf_initialized = true;
                } else {
                    object_state.kf->predict();
                    object_state.kf->update(center_x, center_y);
                }
                // Store the meta pointer for rendering later
                current_frame_meta_map[obj_meta->object_id] = obj_meta;
            }
        }
    }

    // Step 3: Populate messages, render OSD based on state, and prune lost objects
    for (auto it = node->persistent_object_map_.begin(); it != node->persistent_object_map_.end(); )
    {
        auto& object_state = it->second;

        if (object_state.frames_since_seen > KF_LOST_THRESHOLD) {
            if (object_state.id == node->locked_target_id_) node->locked_target_id_ = NO_OBJECT_ID;
            if (object_state.id == node->selected_object_id_) node->selected_object_id_ = NO_OBJECT_ID;
            it = node->persistent_object_map_.erase(it);
            continue;
        }

        if (object_state.frames_since_seen > 0) {
            object_state.kf->predict();
        }

        vision_msgs::msg::Detection2D detection_msg;
        node->populate_ros_detection_message(object_state, detection_msg, current_stamp);
        detection_array_msg.detections.push_back(detection_msg);

        // --- New OSD Rendering Logic ---
        if (node->osd_renderer_) {
            OSDTrackingStatus status = (object_state.frames_since_seen == 0) ? OSDTrackingStatus::DETECTED : OSDTrackingStatus::OCCLUDED;
            NvOSD_RectParams bbox_to_render = object_state.last_bbox;
            if (status == OSDTrackingStatus::OCCLUDED) {
                bbox_to_render.left = object_state.kf->getX() - bbox_to_render.width / 2.0;
                bbox_to_render.top = object_state.kf->getY() - bbox_to_render.height / 2.0;
            }

            // Render locked target with highest priority
            if (object_state.id == node->locked_target_id_) {
                node->osd_renderer_->render_selected_object_osd(
                    batch_meta, (NvDsFrameMeta*)batch_meta->frame_meta_list->data, object_state.id, object_state.class_label,
                    status, true, bbox_to_render, object_state.frames_since_seen,
                    object_state.kf->getVx(), object_state.kf->getVy(), node->camera_fov_rad_
                );
            } 
            // Render selected target (if not also the locked one)
            else if (object_state.id == node->selected_object_id_) {
                node->osd_renderer_->render_selected_object_osd(
                    batch_meta, (NvDsFrameMeta*)batch_meta->frame_meta_list->data, object_state.id, object_state.class_label,
                    status, false, bbox_to_render, object_state.frames_since_seen,
                    object_state.kf->getVx(), object_state.kf->getVy(), node->camera_fov_rad_
                );
            }
            // Render regular, visible objects
            else if (status == OSDTrackingStatus::DETECTED) {
                auto meta_it = current_frame_meta_map.find(object_state.id);
                if (meta_it != current_frame_meta_map.end()) {
                    node->osd_renderer_->render_non_selected_object_osd(batch_meta, (NvDsFrameMeta*)batch_meta->frame_meta_list->data, meta_it->second);
                }
            }
        }
        ++it;
    }

    // Step 4: Publish topics
    if (!detection_array_msg.detections.empty()) {
        node->detection_publisher_->publish(detection_array_msg);
    }

    vision_msgs::msg::Detection2D target_msg;
    auto it = node->persistent_object_map_.find(node->locked_target_id_);
    if (node->locked_target_id_ != NO_OBJECT_ID && it != node->persistent_object_map_.end()) {
        node->populate_ros_detection_message(it->second, target_msg, current_stamp);
    } else {
        target_msg.header.stamp = current_stamp;
        target_msg.header.frame_id = "camera_frame";
        target_msg.id = "-1";
    }
    node->target_publisher_->publish(target_msg);

    return GST_PAD_PROBE_OK;
}


// --- Member Function Implementations ---

void ObjectDetectionNode::handle_lock_target(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);
    if (selected_object_id_ != NO_OBJECT_ID) {
        locked_target_id_ = selected_object_id_;
        RCLCPP_INFO(this->get_logger(), "Target locked: %ld", locked_target_id_);
        response->success = true;
        response->message = "Target locked: " + std::to_string(locked_target_id_);
    } else {
        RCLCPP_WARN(this->get_logger(), "No target selected to lock.");
        response->success = false;
        response->message = "No target selected to lock.";
    }
}

void ObjectDetectionNode::handle_unlock_target(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);
    RCLCPP_INFO(this->get_logger(), "Target unlocked.");
    locked_target_id_ = NO_OBJECT_ID;
    response->success = true;
    response->message = "Target unlocked.";
}

void ObjectDetectionNode::handle_cycle_forward(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    cycle_selected_target(true);
    response->success = true;
    response->message = "Cycled target forward";
}

void ObjectDetectionNode::handle_cycle_backward(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    cycle_selected_target(false);
    response->success = true;
    response->message = "Cycled target backward";
}

void ObjectDetectionNode::populate_ros_detection_message(const TrackedObjectState& object_state, vision_msgs::msg::Detection2D& detection_msg, const rclcpp::Time& stamp)
{
    detection_msg.header.stamp = stamp;
    detection_msg.header.frame_id = "camera_frame";
    
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = object_state.class_label;
    hypothesis.hypothesis.score = (object_state.frames_since_seen == 0) ? object_state.confidence : 0.0;
    detection_msg.results.push_back(hypothesis);

    if (object_state.frames_since_seen == 0) {
        detection_msg.bbox.center.position.x = object_state.last_bbox.left + object_state.last_bbox.width / 2.0;
        detection_msg.bbox.center.position.y = object_state.last_bbox.top + object_state.last_bbox.height / 2.0;
        detection_msg.bbox.size_x = object_state.last_bbox.width;
        detection_msg.bbox.size_y = object_state.last_bbox.height;
    } else {
        detection_msg.bbox.center.position.x = object_state.kf->getX();
        detection_msg.bbox.center.position.y = object_state.kf->getY();
        detection_msg.bbox.size_x = object_state.last_bbox.width;
        detection_msg.bbox.size_y = object_state.last_bbox.height;
    }

    detection_msg.id = std::to_string(object_state.id);
}

void ObjectDetectionNode::cycle_selected_target(bool forward)
{
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);

    if (persistent_object_map_.empty()) {
        if (selected_object_id_ != NO_OBJECT_ID) {
            RCLCPP_INFO(this->get_logger(), "No objects detected. Deselecting target.");
            selected_object_id_ = NO_OBJECT_ID;
        }
        return;
    }

    std::vector<guint64> object_ids;
    for (const auto& pair : persistent_object_map_) {
        object_ids.push_back(pair.first);
    }
    std::sort(object_ids.begin(), object_ids.end());

    auto it = std::find(object_ids.begin(), object_ids.end(), selected_object_id_);

    if (it == object_ids.end()) {
        selected_object_id_ = forward ? object_ids.front() : object_ids.back();
        RCLCPP_INFO(this->get_logger(), "No object selected. Selecting first/last: %lu", selected_object_id_);
    } else {
        if (forward) {
            it++;
            selected_object_id_ = (it == object_ids.end()) ? NO_OBJECT_ID : *it;
        } else {
            selected_object_id_ = (it == object_ids.begin()) ? NO_OBJECT_ID : *(--it);
        }
    }

    if (selected_object_id_ == NO_OBJECT_ID) {
        RCLCPP_INFO(this->get_logger(), "Cycled to deselection.");
    } else {
        RCLCPP_INFO(this->get_logger(), "Cycled to new object: %lu", selected_object_id_);
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
