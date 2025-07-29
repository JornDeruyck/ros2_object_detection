#include "ros2_object_detection/object_detection.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include "gstnvdsmeta.h"

// ROS messages
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/opencv.hpp>

// === Appsink callback: get frames, compress, publish ===
GstFlowReturn ObjectDetectionNode::new_sample_callback(GstElement *sink, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstSample *sample = nullptr;
    g_signal_emit_by_name(sink, "pull-sample", &sample);

    if (!sample)
        return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;

    if (gst_buffer_map(buffer, &map, GST_MAP_READ))
    {
        // Publish directly as JPEG compressed image (no decoding or re-encoding)
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


// === Probe callback: extract detections from metadata ===
GstPadProbeReturn ObjectDetectionNode::osd_probe_callback(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstBuffer *gst_buffer = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);

    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header.stamp = node->get_clock()->now();
    detection_array_msg.header.frame_id = "camera_frame";

    for (GList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;

            vision_msgs::msg::Detection2D detection;
            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;

            hypothesis.hypothesis.class_id = obj_meta->obj_label;
            hypothesis.hypothesis.score = obj_meta->confidence;
            detection.results.push_back(hypothesis);

            NvOSD_RectParams &rect_params = obj_meta->rect_params;
            detection.bbox.center.position.x = rect_params.left + rect_params.width / 2.0;
            detection.bbox.center.position.y = rect_params.top + rect_params.height / 2.0;
            detection.bbox.size_x = rect_params.width;
            detection.bbox.size_y = rect_params.height;

            detection_array_msg.detections.push_back(detection);
        }
    }

    if (!detection_array_msg.detections.empty())
    {
        node->detection_publisher_->publish(detection_array_msg);
    }

    return GST_PAD_PROBE_OK;
}

// === Constructor ===
ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions &options)
    : Node("object_detection_node", options)
{
    // Load pipeline config
    auto config_path = this->declare_parameter<std::string>("pipeline_config");
    YAML::Node config = YAML::LoadFile(config_path);
    std::string pipeline_str = config["pipeline"].as<std::string>();

    // QoS tuned for minimal buffering
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
    qos.reliable();          // instead of best_effort
    qos.durability_volatile();

    // Create publishers
    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", qos);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", qos);

    // Init GStreamer
    gst_init(nullptr, nullptr);
    main_loop_ = g_main_loop_new(nullptr, FALSE);
    GError *error = nullptr;

    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline_)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to parse pipeline: %s", error ? error->message : "Unknown error");
        if (error) g_error_free(error);
        return;
    }

    // Hook OSD metadata probe
    GstElement *osd = gst_bin_get_by_name(GST_BIN(pipeline_), "nvdsosd_0");
    if (osd)
    {
        GstPad *sinkpad = gst_element_get_static_pad(osd, "sink");
        gst_pad_add_probe(sinkpad, GST_PAD_PROBE_TYPE_BUFFER, osd_probe_callback, this, nullptr);
        gst_object_unref(sinkpad);
        gst_object_unref(osd);
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find nvdsosd_0 in pipeline");
    }

    // Hook appsink for frames
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline_), "ros_sink");
    if (appsink)
    {
        g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample_callback), this);
        gst_object_unref(appsink);
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find ros_sink in pipeline");
    }

    // Start pipeline
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    RCLCPP_INFO(this->get_logger(), "Starting GStreamer main loop...");
    gstreamer_thread_ = std::thread([this]() { g_main_loop_run(main_loop_); });
}

// === Destructor ===
ObjectDetectionNode::~ObjectDetectionNode()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down pipeline...");
    if (main_loop_)
    {
        g_main_loop_quit(main_loop_);
    }
    if (gstreamer_thread_.joinable())
    {
        gstreamer_thread_.join();
    }
    if (pipeline_)
    {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }
    if (main_loop_)
    {
        g_main_loop_unref(main_loop_);
    }
}

// === Main entrypoint ===
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<ObjectDetectionNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
