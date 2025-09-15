// src/object_detection.cpp
#include "ros2_object_detection/object_detection.hpp"
#include "ros2_object_detection/osd_renderer.hpp"
#include "ros2_object_detection/constants.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <cmath>
#include <cstring>
#include <vector>
#include <cinttypes> 

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <glib.h>

#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "nvll_osd_struct.h"
#include "nvds_tracker_meta.h"

#include "sensor_msgs/msg/compressed_image.hpp"
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

// --- Class Constructor and Destructor ---

ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions &options)
    : Node("object_detection_node", options),
      pipeline_(nullptr), main_loop_(nullptr),
      selected_object_id_(NO_OBJECT_ID), locked_target_id_(NO_OBJECT_ID)
{
    RCLCPP_INFO(this->get_logger(), "Initializing ObjectDetectionNode...");

    this->declare_parameter<std::string>("pipeline_string", "");
    this->declare_parameter<std::vector<long int>>("allowed_class_ids", std::vector<long int>());
    this->declare_parameter<bool>("use_qos_reliable", true);
    this->declare_parameter<int>("qos_history_depth", 1);
    this->declare_parameter<double>("camera_fov", 90.0);
    this->declare_parameter<std::string>("frame_id", "camera_frame");
    this->declare_parameter<std::string>("osd_element_name", "nvdsosd0");
    this->declare_parameter<std::string>("appsink_element_name", "ros_sink");
    this->declare_parameter<int>("kf_lost_threshold", 30);
    this->declare_parameter<double>("latency_smoothing_alpha", 0.05);
    this->declare_parameter<bool>("enable_latency_measurement", false);

    std::string pipeline_string = this->get_parameter("pipeline_string").as_string();
    allowed_class_ids_ = this->get_parameter("allowed_class_ids").as_integer_array();
    bool use_qos_reliable = this->get_parameter("use_qos_reliable").as_bool();
    int qos_history_depth = this->get_parameter("qos_history_depth").as_int();
    double camera_fov_deg = this->get_parameter("camera_fov").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();
    osd_element_name_ = this->get_parameter("osd_element_name").as_string();
    appsink_element_name_ = this->get_parameter("appsink_element_name").as_string();
    kf_lost_threshold_ = this->get_parameter("kf_lost_threshold").as_int();
    latency_smoothing_alpha_ = this->get_parameter("latency_smoothing_alpha").as_double();
    enable_latency_measurement_ = this->get_parameter("enable_latency_measurement").as_bool();
    
    camera_fov_rad_ = camera_fov_deg * M_PI / 180.0;

    if (pipeline_string.empty()) {
        RCLCPP_FATAL(this->get_logger(), "Parameter 'pipeline_string' is empty.");
        throw std::runtime_error("Empty 'pipeline_string' parameter.");
    }

    rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(qos_history_depth));
    qos_profile.reliability(use_qos_reliable ? RMW_QOS_POLICY_RELIABILITY_RELIABLE : RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    qos_profile.durability_volatile();

    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", qos_profile);
    target_publisher_ = this->create_publisher<vision_msgs::msg::Detection2D>("~/target", qos_profile);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("~/image_compressed", qos_profile);
    osd_renderer_ = std::make_unique<OSDRenderer>(this);

    lock_target_service_ = this->create_service<std_srvs::srv::Trigger>("~/lock_target", std::bind(&ObjectDetectionNode::handle_lock_target, this, std::placeholders::_1, std::placeholders::_2));
    unlock_target_service_ = this->create_service<std_srvs::srv::Trigger>("~/unlock_target", std::bind(&ObjectDetectionNode::handle_unlock_target, this, std::placeholders::_1, std::placeholders::_2));
    cycle_target_forward_service_ = this->create_service<std_srvs::srv::Trigger>("~/cycle_target_forward", std::bind(&ObjectDetectionNode::handle_cycle_forward, this, std::placeholders::_1, std::placeholders::_2));
    cycle_target_backward_service_ = this->create_service<std_srvs::srv::Trigger>("~/cycle_target_backward", std::bind(&ObjectDetectionNode::handle_cycle_backward, this, std::placeholders::_1, std::placeholders::_2));

    gst_init(nullptr, nullptr);
    main_loop_ = g_main_loop_new(nullptr, FALSE);
    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_string.c_str(), &error);
    if (!pipeline_) {
        RCLCPP_FATAL(this->get_logger(), "Failed to parse GStreamer pipeline: %s", error ? error->message : "Unknown error");
        if (error) g_error_free(error);
        throw std::runtime_error("GStreamer pipeline parsing failed.");
    }

    GstBus *bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, bus_callback, this);
    gst_object_unref(bus);

    GstElement *osd_element = gst_bin_get_by_name(GST_BIN(pipeline_), osd_element_name_.c_str());
    if (osd_element) {
        GstPad *osd_sink_pad = gst_element_get_static_pad(osd_element, "sink");
        if (osd_sink_pad) {
            gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_probe_callback, this, nullptr);
            gst_object_unref(osd_sink_pad);
        } else {
            RCLCPP_WARN(this->get_logger(), "Could not get sink pad for OSD element '%s'", osd_element_name_.c_str());
        }
        gst_object_unref(osd_element);
    } else {
        RCLCPP_ERROR(this->get_logger(), "Could not find OSD element '%s' in the pipeline.", osd_element_name_.c_str());
    }

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline_), appsink_element_name_.c_str());
    if (appsink) {
        g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample_callback), this);
        gst_object_unref(appsink);
    } else {
        RCLCPP_WARN(this->get_logger(), "Could not find AppSink element '%s'. No compressed image will be published.", appsink_element_name_.c_str());
    }

    if (enable_latency_measurement_) {
        RCLCPP_INFO(this->get_logger(), "Enabling GStreamer latency probes.");
        g_signal_connect(pipeline_, "element-added", G_CALLBACK(element_added_callback), this);
        add_latency_probes(GST_BIN(pipeline_));
    }

    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    gstreamer_thread_ = std::thread([this]() { g_main_loop_run(main_loop_); });
    RCLCPP_INFO(this->get_logger(), "ObjectDetectionNode fully initialized and pipeline is playing.");
}


ObjectDetectionNode::~ObjectDetectionNode()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down ObjectDetectionNode...");
    if (main_loop_ && g_main_loop_is_running(main_loop_)) {
        g_main_loop_quit(main_loop_);
    }
    if (gstreamer_thread_.joinable()) {
        gstreamer_thread_.join();
    }
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }
    if (main_loop_) {
        g_main_loop_unref(main_loop_);
    }
    RCLCPP_INFO(this->get_logger(), "ObjectDetectionNode shut down complete.");
}

// --- GStreamer Bus Watch ---
gboolean ObjectDetectionNode::bus_callback(GstBus * /*bus*/, GstMessage *msg, gpointer data) {
    auto *node = static_cast<ObjectDetectionNode *>(data);
    node->handle_bus_message(msg);
    return TRUE;
}
void ObjectDetectionNode::handle_bus_message(GstMessage *msg) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError *err = nullptr;
            gchar *debug_info = nullptr;
            gst_message_parse_error(msg, &err, &debug_info);
            RCLCPP_FATAL(this->get_logger(), "GStreamer Error from %s: %s", GST_OBJECT_NAME(msg->src), err->message);
            RCLCPP_FATAL(this->get_logger(), "Debugging info: %s", debug_info ? debug_info : "none");
            g_error_free(err);
            g_free(debug_info);
            g_main_loop_quit(main_loop_);
            break;
        }
        case GST_MESSAGE_EOS:
            RCLCPP_INFO(this->get_logger(), "GStreamer: End-Of-Stream reached.");
            g_main_loop_quit(main_loop_);
            break;
        default:
            break;
    }
}

// --- GStreamer Callbacks ---
GstFlowReturn ObjectDetectionNode::new_sample_callback(GstElement *sink, gpointer user_data) {
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
        msg.header.frame_id = node->frame_id_;
        msg.format = "jpeg";
        msg.data.assign(map.data, map.data + map.size);
        node->compressed_publisher_->publish(msg);
        gst_buffer_unmap(buffer, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

GstPadProbeReturn ObjectDetectionNode::osd_probe_callback(GstPad * /*pad*/, GstPadProbeInfo *info, gpointer user_data) {
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstBuffer *gst_buffer = GST_BUFFER(info->data);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    const rclcpp::Time current_stamp = node->get_clock()->now();
    std::vector<TrackedObjectState> objects_to_process; 
    guint64 current_locked_id = NO_OBJECT_ID;
    guint64 current_selected_id = NO_OBJECT_ID;

    {
        std::lock_guard<std::mutex> lock(node->tracked_objects_mutex_);
        node->update_tracking_state(batch_meta);
        node->prune_lost_tracks();
        for (const auto& pair : node->persistent_object_map_) {
            objects_to_process.push_back(pair.second);
        }
        current_locked_id = node->locked_target_id_;
        current_selected_id = node->selected_object_id_;
    }

    node->render_osd(batch_meta, objects_to_process, current_locked_id, current_selected_id);
    node->publish_messages(objects_to_process, current_locked_id, current_stamp);
    if (node->enable_latency_measurement_) {
        node->calculate_and_clean_latency(gst_buffer);
    }
    return GST_PAD_PROBE_OK;
}

// --- Core Logic ---
void ObjectDetectionNode::update_tracking_state(NvDsBatchMeta* batch_meta) {
    for (auto& pair : persistent_object_map_) {
        pair.second.frames_since_seen++;
    }
    for (GList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) continue;
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue;
            bool is_allowed_class = allowed_class_ids_.empty() || (std::find(allowed_class_ids_.begin(), allowed_class_ids_.end(), obj_meta->class_id) != allowed_class_ids_.end());
            if (is_allowed_class) {
                auto& object_state = persistent_object_map_[obj_meta->object_id];
                if (!object_state.kf_initialized) {
                    object_state.kf = std::make_shared<KalmanFilter2D>();
                }
                object_state.id = obj_meta->object_id;
                object_state.class_label = std::string(obj_meta->obj_label);
                object_state.confidence = obj_meta->confidence;
                object_state.last_bbox = bbox_coords_to_rect_params(obj_meta->tracker_bbox_info.org_bbox_coords);
                object_state.frames_since_seen = 0;
                double center_x = object_state.last_bbox.left + object_state.last_bbox.width / 2.0;
                double center_y = object_state.last_bbox.top + object_state.last_bbox.height / 2.0;
                if (!object_state.kf_initialized) {
                    object_state.kf->init(center_x, center_y);
                    object_state.kf_initialized = true;
                } else {
                    object_state.kf->predict();
                    object_state.kf->update(center_x, center_y);
                }
            }
        }
    }
}
void ObjectDetectionNode::prune_lost_tracks() {
    for (auto it = persistent_object_map_.begin(); it != persistent_object_map_.end(); ) {
        if (it->second.frames_since_seen > kf_lost_threshold_) {
            if (it->first == static_cast<guint64>(locked_target_id_)) locked_target_id_ = NO_OBJECT_ID;
            if (it->first == static_cast<guint64>(selected_object_id_)) selected_object_id_ = NO_OBJECT_ID;
            it = persistent_object_map_.erase(it);
        } else {
            ++it;
        }
    }
}
void ObjectDetectionNode::render_osd(NvDsBatchMeta* batch_meta, const std::vector<TrackedObjectState>& objects_to_render, guint64 locked_id, guint64 selected_id) {
    if (!osd_renderer_ || !batch_meta->frame_meta_list) return;
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta*)batch_meta->frame_meta_list->data;
    
    std::map<guint64, NvDsObjectMeta*> current_meta_map;
    for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
        if (obj_meta) {
            current_meta_map[obj_meta->object_id] = obj_meta;
            obj_meta->rect_params.border_width = 0;
            obj_meta->rect_params.has_bg_color = 0;
            if (obj_meta->text_params.display_text) {
                g_free(obj_meta->text_params.display_text);
                obj_meta->text_params.display_text = nullptr;
            }
        }
    }

    double center_x = frame_meta->source_frame_width / 2.0;
    double center_y = frame_meta->source_frame_height / 2.0;
    osd_renderer_->update_and_display_fps(batch_meta, frame_meta);
    if (enable_latency_measurement_ && !smoothed_latency_map_.empty()) {
        osd_renderer_->display_latency(batch_meta, frame_meta, smoothed_latency_map_);
    }
    osd_renderer_->draw_reticule(batch_meta, frame_meta, center_x, center_y, 50.0, osd_renderer_->white_color_, 2, ReticuleStyle::CROSS_GAP);

    for (const auto& object_state : objects_to_render) {
        OSDTrackingStatus status = (object_state.frames_since_seen == 0) ? OSDTrackingStatus::DETECTED : OSDTrackingStatus::OCCLUDED;
        NvOSD_RectParams bbox_to_render = object_state.last_bbox;
        
        if (object_state.kf) {
            if (status == OSDTrackingStatus::OCCLUDED) {
                object_state.kf->predict();
                bbox_to_render.left = object_state.kf->getX() - bbox_to_render.width / 2.0;
                bbox_to_render.top = object_state.kf->getY() - bbox_to_render.height / 2.0;
            }
            bool is_locked = (object_state.id == locked_id);
            if (is_locked || object_state.id == selected_id) {
                osd_renderer_->render_selected_object_osd(batch_meta, frame_meta, object_state.id, object_state.class_label,
                    status, is_locked, bbox_to_render, object_state.frames_since_seen,
                    object_state.kf->getVx(), object_state.kf->getVy(), camera_fov_rad_);
            } else if (status == OSDTrackingStatus::DETECTED) {
                auto meta_it = current_meta_map.find(object_state.id);
                if (meta_it != current_meta_map.end()) {
                    osd_renderer_->render_non_selected_object_osd(batch_meta, frame_meta, meta_it->second);
                }
            }
        }
    }
}
void ObjectDetectionNode::publish_messages(const std::vector<TrackedObjectState>& objects_to_render, guint64 locked_id, const rclcpp::Time& stamp) {
    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header.stamp = stamp;
    detection_array_msg.header.frame_id = frame_id_;
    const TrackedObjectState* locked_target_state = nullptr;
    for (const auto& object_state : objects_to_render) {
        vision_msgs::msg::Detection2D detection_msg;
        populate_ros_detection_message(object_state, detection_msg, stamp);
        detection_array_msg.detections.push_back(detection_msg);
        if (object_state.id == locked_id) {
            locked_target_state = &object_state;
        }
    }
    if (!detection_array_msg.detections.empty()) {
        detection_publisher_->publish(detection_array_msg);
    }
    vision_msgs::msg::Detection2D target_msg;
    if (locked_target_state) {
        populate_ros_detection_message(*locked_target_state, target_msg, stamp);
    } else {
        target_msg.header.stamp = stamp;
        target_msg.header.frame_id = frame_id_;
        target_msg.id = "-1";
    }
    target_publisher_->publish(target_msg);
}

// --- Latency Measurement ---
void ObjectDetectionNode::calculate_and_clean_latency(GstBuffer *gst_buffer) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    auto latency_it = latency_map_.find(gst_buffer);
    if (latency_it != latency_map_.end()) {
        const auto& timestamps = latency_it->second;
        for (auto const& [key, val] : timestamps) {
            if (key.find("_sink") != std::string::npos) {
                std::string base_name = key.substr(0, key.find("_sink"));
                auto src_it = timestamps.find(base_name + "_src");
                if (src_it != timestamps.end()) {
                    std::chrono::duration<double, std::milli> ms = val - src_it->second;
                    double current_latency = std::abs(ms.count());
                    auto smooth_it = smoothed_latency_map_.find(base_name);
                    if (smooth_it != smoothed_latency_map_.end()) {
                        smooth_it->second = (latency_smoothing_alpha_ * current_latency) + (1.0 - latency_smoothing_alpha_) * smooth_it->second;
                    } else {
                        smoothed_latency_map_[base_name] = current_latency;
                    }
                }
            }
        }
        gst_buffer_unref(gst_buffer);
        latency_map_.erase(latency_it);
    }
}
GstPadProbeReturn ObjectDetectionNode::latency_probe_sink(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstBuffer *buf = GST_BUFFER(info->data);
    
    GstElement *parent_element = gst_pad_get_parent_element(pad);
    std::string element_name = gst_element_get_name(parent_element);
    gst_object_unref(parent_element);
    
    std::lock_guard<std::mutex> lock(node->latency_mutex_);
    gst_buffer_ref(buf);
    node->latency_map_[buf][element_name + "_sink"] = std::chrono::steady_clock::now();
    
    return GST_PAD_PROBE_OK;
}
GstPadProbeReturn ObjectDetectionNode::latency_probe_src(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstBuffer *buf = GST_BUFFER(info->data);
    
    GstElement *parent_element = gst_pad_get_parent_element(pad);
    std::string element_name = gst_element_get_name(parent_element);
    gst_object_unref(parent_element);
    
    std::lock_guard<std::mutex> lock(node->latency_mutex_);
    node->latency_map_[buf][element_name + "_src"] = std::chrono::steady_clock::now();

    return GST_PAD_PROBE_OK;
}

// --- Service Handlers and Helpers ---
void ObjectDetectionNode::handle_lock_target(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/, std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);
    if (selected_object_id_ != NO_OBJECT_ID) {
        locked_target_id_ = selected_object_id_;
        // FIX: Use correct C-style format specifiers for logging
        RCLCPP_INFO(this->get_logger(), "Target locked: %ld", locked_target_id_);
        response->success = true;
        response->message = "Target locked: " + std::to_string(locked_target_id_);
    } else {
        RCLCPP_WARN(this->get_logger(), "No target selected to lock.");
        response->success = false;
        response->message = "No target selected to lock.";
    }
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

    auto it = persistent_object_map_.find(selected_object_id_);

    if (it == persistent_object_map_.end()) {
        if (forward) {
            selected_object_id_ = persistent_object_map_.begin()->first;
        } else {
            auto last_it = std::prev(persistent_object_map_.end());
            selected_object_id_ = last_it->first;
        }
        // FIX: Use correct C-style format specifiers for logging
        RCLCPP_INFO(this->get_logger(), "No object selected. Selecting first/last: %ld", selected_object_id_);
    } else {
        if (forward) {
            it++;
            selected_object_id_ = (it == persistent_object_map_.end()) ? NO_OBJECT_ID : it->first;
        } else {
            if (it == persistent_object_map_.begin()) {
                selected_object_id_ = NO_OBJECT_ID;
            } else {
                it--;
                selected_object_id_ = it->first;
            }
        }
    }

    if (selected_object_id_ == NO_OBJECT_ID) {
        RCLCPP_INFO(this->get_logger(), "Cycled to deselection.");
    } else {
        // FIX: Use correct C-style format specifiers for logging
        RCLCPP_INFO(this->get_logger(), "Cycled to new object: %ld", selected_object_id_);
    }
}

void ObjectDetectionNode::element_added_callback(GstBin * /*bin*/, GstElement *element, gpointer user_data) {
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    const gchar* name = gst_element_get_name(element);
    RCLCPP_DEBUG(node->get_logger(), "Element added: %s", name);
    GstPad *sinkpad = gst_element_get_static_pad(element, "sink");
    if (sinkpad) {
        gst_pad_add_probe(sinkpad, GST_PAD_PROBE_TYPE_BUFFER, latency_probe_sink, node, nullptr);
        gst_object_unref(sinkpad);
    }
}
void ObjectDetectionNode::add_latency_probes(GstBin *bin) {
    GstIterator *it = gst_bin_iterate_elements(bin);
    GValue item = G_VALUE_INIT;
    bool done = false;

    while (!done) {
        switch (gst_iterator_next(it, &item)) {
            case GST_ITERATOR_OK: {
                GstElement *element = GST_ELEMENT(g_value_get_object(&item));
                if (GST_IS_BIN(element)) {
                    add_latency_probes(GST_BIN(element));
                } else {
                    GstPad *sinkpad = gst_element_get_static_pad(element, "sink");
                    if (sinkpad) {
                        gst_pad_add_probe(sinkpad, GST_PAD_PROBE_TYPE_BUFFER, latency_probe_sink, this, nullptr);
                        gst_object_unref(sinkpad);
                    }
                    GstPad *srcpad = gst_element_get_static_pad(element, "src");
                    if (srcpad) {
                        gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_BUFFER, latency_probe_src, this, nullptr);
                        gst_object_unref(srcpad);
                    }
                }
                g_value_unset(&item);
                break;
            }
            case GST_ITERATOR_RESYNC:
                gst_iterator_resync(it);
                break;
            case GST_ITERATOR_DONE:
                done = true;
                break;
            case GST_ITERATOR_ERROR:
                // Handle error case to satisfy -Wswitch warning
                done = true;
                break;
        }
    }
    gst_iterator_free(it);
}
void ObjectDetectionNode::handle_unlock_target(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/, std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);
    RCLCPP_INFO(this->get_logger(), "Target unlocked.");
    locked_target_id_ = NO_OBJECT_ID;
    response->success = true;
    response->message = "Target unlocked.";
}
void ObjectDetectionNode::handle_cycle_forward(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/, std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    cycle_selected_target(true);
    response->success = true;
    response->message = "Cycled target forward";
}
void ObjectDetectionNode::handle_cycle_backward(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/, std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    cycle_selected_target(false);
    response->success = true;
    response->message = "Cycled target backward";
}
void ObjectDetectionNode::populate_ros_detection_message(const TrackedObjectState& object_state, vision_msgs::msg::Detection2D& detection_msg, const rclcpp::Time& stamp) {
    detection_msg.header.stamp = stamp;
    detection_msg.header.frame_id = frame_id_;
    
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = object_state.class_label;
    hypothesis.hypothesis.score = (object_state.frames_since_seen == 0) ? object_state.confidence : 0.0;
    detection_msg.results.push_back(hypothesis);

    if (object_state.kf) {
        if (object_state.frames_since_seen == 0) {
            detection_msg.bbox.center.position.x = object_state.last_bbox.left + object_state.last_bbox.width / 2.0;
            detection_msg.bbox.center.position.y = object_state.last_bbox.top + object_state.last_bbox.height / 2.0;
        } else {
            detection_msg.bbox.center.position.x = object_state.kf->getX();
            detection_msg.bbox.center.position.y = object_state.kf->getY();
        }
    }
    detection_msg.bbox.size_x = object_state.last_bbox.width;
    detection_msg.bbox.size_y = object_state.last_bbox.height;
    detection_msg.id = std::to_string(object_state.id);
}
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    try {
        auto node = std::make_shared<ObjectDetectionNode>(options);
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Node initialization failed: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}