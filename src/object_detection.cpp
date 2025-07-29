#include "ros2_object_detection/object_detection.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
// DeepStream metadata headers
#include "nvdsmeta.h"     // For NvDsDisplayMeta and other core metadata structures
#include "gstnvdsmeta.h"  // For gst_buffer_get_nvds_batch_meta
#include "nvll_osd_struct.h" // For NvOSD_TextParams, NvOSD_RectParams, NvOSD_LineParams etc.
#include <chrono>         // For FPS calculation
#include <mutex>          // For FPS mutex
#include <algorithm>      // For std::find
#include <string>         // For std::string
#include <iomanip>        // For std::setprecision
#include <sstream>        // For std::stringstream

// ROS messages
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/opencv.hpp> // Not explicitly used for OSD, but good to keep if needed for image processing

// === Appsink callback: get frames, compress, publish ===
// This callback is triggered when a new sample (frame) is available in the appsink.
// It pulls the sample, maps its buffer, and publishes it as a compressed JPEG image.
GstFlowReturn ObjectDetectionNode::new_sample_callback(GstElement *sink, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstSample *sample = nullptr;
    // Pull the sample from the appsink
    g_signal_emit_by_name(sink, "pull-sample", &sample);

    if (!sample)
        return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;

    // Map the buffer to access its data
    if (gst_buffer_map(buffer, &map, GST_MAP_READ))
    {
        // Create a CompressedImage ROS message
        auto msg = sensor_msgs::msg::CompressedImage();
        msg.header.stamp = node->get_clock()->now();
        msg.header.frame_id = "camera_frame"; // Set appropriate frame ID
        msg.format = "jpeg";                  // Indicate the image format
        // Copy the raw JPEG data directly to the message
        msg.data.assign(map.data, map.data + map.size);

        // Publish the compressed image
        node->compressed_publisher_->publish(msg);

        gst_buffer_unmap(buffer, &map);
    }

    // Unreference the sample to release resources
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}


// === Probe callback: extract detections from metadata and add OSD overlays ===
// This callback is attached to the sink pad of the nvdsosd element.
// It processes DeepStream metadata for object detections and adds custom OSD overlays
// for FPS, object confidence, bounding boxes, labels, highlighting, and reticule.
GstPadProbeReturn ObjectDetectionNode::osd_probe_callback(GstPad * /*pad*/, GstPadProbeInfo *info, gpointer user_data)
{
    auto *node = static_cast<ObjectDetectionNode *>(user_data);
    GstBuffer *gst_buffer = (GstBuffer *)info->data;
    // Retrieve the batch metadata from the GStreamer buffer
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);

    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header.stamp = node->get_clock()->now();
    detection_array_msg.header.frame_id = "camera_frame";

    // --- FPS Calculation ---
    // Lock mutex to protect FPS variables during update
    std::lock_guard<std::mutex> lock_fps(node->fps_mutex_); // Use different lock name

    node->frame_counter_++;

    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = now - node->last_fps_update_time_;

    if (elapsed_seconds.count() >= 1.0 || node->frame_counter_ >= 30)
    {
        node->current_fps_display_ = node->frame_counter_ / elapsed_seconds.count();
        node->frame_counter_ = 0;
        node->last_fps_update_time_ = now;
    }

    // Lock mutex for tracked objects
    std::lock_guard<std::mutex> lock_objects(node->tracked_objects_mutex_); // Use different lock name
    node->current_tracked_objects_.clear(); // Clear objects from previous frame

    // Iterate through each frame in the batch (though this pipeline has batch-size=1)
    for (GList *l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        // --- Add FPS OSD overlay (using a new NvDsDisplayMeta) ---
        NvDsDisplayMeta *fps_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (fps_display_meta)
        {
            fps_display_meta->num_labels = 1;
            fps_display_meta->text_params[0].display_text = g_strdup_printf("FPS: %.2f", node->current_fps_display_);
            fps_display_meta->text_params[0].x_offset = 10;
            fps_display_meta->text_params[0].y_offset = 10;
            fps_display_meta->text_params[0].font_params.font_name = (gchar *)"Sans";
            fps_display_meta->text_params[0].font_params.font_size = 14;
            fps_display_meta->text_params[0].font_params.font_color.red = 1.0;
            fps_display_meta->text_params[0].font_params.font_color.green = 1.0;
            fps_display_meta->text_params[0].font_params.font_color.blue = 1.0;
            fps_display_meta->text_params[0].font_params.font_color.alpha = 1.0;
            fps_display_meta->text_params[0].set_bg_clr = 0; // No background box

            nvds_add_display_meta_to_frame(frame_meta, fps_display_meta);
        }

        // --- Process and Modify existing NvDsObjectMeta for OSD ---
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;

            // Store object's current bounding box for selection logic
            node->current_tracked_objects_[obj_meta->object_id] = obj_meta->rect_params;

            // Populate ROS Detection2D message
            vision_msgs::msg::Detection2D detection;
            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = obj_meta->obj_label;
            hypothesis.hypothesis.score = obj_meta->confidence;
            detection.results.push_back(hypothesis);
            detection.bbox.center.position.x = obj_meta->rect_params.left + obj_meta->rect_params.width / 2.0;
            detection.bbox.center.position.y = obj_meta->rect_params.top + obj_meta->rect_params.height / 2.0;
            detection.bbox.size_x = obj_meta->rect_params.width;
            detection.bbox.size_y = obj_meta->rect_params.height;
            detection_array_msg.detections.push_back(detection);

            // --- Customize Bounding Box (obj_meta->rect_params) ---
            obj_meta->rect_params.border_width = 3; // Thicker border
            obj_meta->rect_params.has_bg_color = 0; // No background fill for box

            // Set colors based on selection
            if (obj_meta->object_id == node->selected_object_id_)
            {
                // Selected object: Red
                obj_meta->rect_params.border_color.red = 1.0;
                obj_meta->rect_params.border_color.green = 0.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_color.alpha = 1.0;
            }
            else
            {
                // Other objects: Blue
                obj_meta->rect_params.border_color.red = 0.0;
                obj_meta->rect_params.border_color.green = 0.0;
                obj_meta->rect_params.border_color.blue = 1.0;
                obj_meta->rect_params.border_color.alpha = 1.0;
            }

            // --- Customize Object Label and Confidence (obj_meta->text_params) ---
            // Concatenate label, ID, and confidence into a single string
            std::stringstream ss_label_conf;
            ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id << " Conf: " << std::fixed << std::setprecision(2) << obj_meta->confidence;
            std::string label_conf_str = ss_label_conf.str();

            // Free previous display_text if it was allocated by g_strdup
            if (obj_meta->text_params.display_text) {
                g_free(obj_meta->text_params.display_text);
            }
            obj_meta->text_params.display_text = g_strdup(label_conf_str.c_str());

            // Position text "on top" of the bounding box with more clearance
            obj_meta->text_params.x_offset = (guint)(obj_meta->rect_params.left);
            obj_meta->text_params.y_offset = (guint)(obj_meta->rect_params.top - 25); // Increased offset to move text further up
            // If text would go off-screen, push it below the box
            if ((gint)obj_meta->text_params.y_offset < 0) {
                obj_meta->text_params.y_offset = (guint)(obj_meta->rect_params.top + obj_meta->rect_params.height + 5);
            }

            obj_meta->text_params.font_params.font_name = (gchar *)"Sans";
            obj_meta->text_params.font_params.font_size = 12;
            obj_meta->text_params.font_params.font_color = obj_meta->rect_params.border_color; // Same color as bounding box
            obj_meta->text_params.set_bg_clr = 0; // No background box
        }

        // --- Draw Reticule for Selected Object (using a new NvDsDisplayMeta) ---
        if (node->selected_object_id_ != UNTRACKED_OBJECT_ID)
        {
            auto it = node->current_tracked_objects_.find(node->selected_object_id_);
            if (it != node->current_tracked_objects_.end())
            {
                NvOSD_RectParams &selected_rect = it->second;
                gfloat center_x = selected_rect.left + selected_rect.width / 2.0;
                gfloat center_y = selected_rect.top + selected_rect.height / 2.0;
                gfloat reticule_size = 20.0; // Size of the crosshair arms

                NvDsDisplayMeta *reticule_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                if (reticule_display_meta)
                {
                    reticule_display_meta->num_lines = 2; // Two lines for crosshair

                    // Reticule color (Red, matching selected object)
                    NvOSD_ColorParams reticule_color = {1.0, 0.0, 0.0, 1.0};

                    // Horizontal line
                    reticule_display_meta->line_params[0].x1 = (guint)(center_x - reticule_size / 2.0);
                    reticule_display_meta->line_params[0].y1 = (guint)center_y;
                    reticule_display_meta->line_params[0].x2 = (guint)(center_x + reticule_size / 2.0);
                    reticule_display_meta->line_params[0].y2 = (guint)center_y;
                    reticule_display_meta->line_params[0].line_width = 2;
                    reticule_display_meta->line_params[0].line_color = reticule_color;

                    // Vertical line
                    reticule_display_meta->line_params[1].x1 = (guint)center_x;
                    reticule_display_meta->line_params[1].y1 = (guint)(center_y - reticule_size / 2.0);
                    reticule_display_meta->line_params[1].x2 = (guint)center_x;
                    reticule_display_meta->line_params[1].y2 = (guint)(center_y + reticule_size / 2.0);
                    reticule_display_meta->line_params[1].line_width = 2;
                    reticule_display_meta->line_params[1].line_color = reticule_color;

                    nvds_add_display_meta_to_frame(frame_meta, reticule_display_meta);
                }
            } else {
                // If selected object is no longer tracked, reset selection
                node->selected_object_id_ = UNTRACKED_OBJECT_ID;
            }
        }
    }

    // Publish the array of detections if any objects were detected
    if (!detection_array_msg.detections.empty())
    {
        node->detection_publisher_->publish(detection_array_msg);
    }

    return GST_PAD_PROBE_OK;
}

// === Constructor ===
// Initializes the ROS2 node, loads the GStreamer pipeline, and sets up callbacks.
ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions &options)
    : Node("object_detection_node", options),
      // Initialize FPS calculation members
      last_fps_update_time_(std::chrono::steady_clock::now()), // Set initial time
      frame_counter_(0),                                       // Reset frame counter
      current_fps_display_(0.0),                               // Initialize displayed FPS to zero
      selected_object_id_(UNTRACKED_OBJECT_ID)                 // Initialize selected object to untracked
{
    // Declare and load pipeline configuration from ROS parameter
    auto config_path = this->declare_parameter<std::string>("pipeline_config");
    YAML::Node config = YAML::LoadFile(config_path);
    std::string pipeline_str = config["pipeline"].as<std::string>();

    // Define QoS settings for publishers to minimize buffering and latency
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)); // Keep only the latest message
    qos.reliable();                              // Ensure delivery (vs. best_effort)
    qos.durability_volatile();                   // Messages are not persistent

    // Create ROS publishers for detections and compressed images
    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", qos);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", qos);

    // Init GStreamer
    gst_init(nullptr, nullptr);
    main_loop_ = g_main_loop_new(nullptr, FALSE); // Create a new GLib main loop

    GError *error = nullptr;

    // Parse the GStreamer pipeline string from YAML
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline_)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to parse pipeline: %s", error ? error->message : "Unknown error");
        if (error) g_error_free(error);
        return;
    }

    // Hook probe to the sink pad of 'nvdsosd_0'
    // This ensures that all inference and tracking metadata is available,
    // and the custom OSD will be rendered by nvdsosd.
    GstElement *osd_element = gst_bin_get_by_name(GST_BIN(pipeline_), "nvdsosd_0");
    if (osd_element)
    {
        GstPad *osd_sink_pad = gst_element_get_static_pad(osd_element, "sink");
        if (osd_sink_pad) {
            gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_probe_callback, this, nullptr);
            gst_object_unref(osd_sink_pad);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Could not get sink pad from nvdsosd_0 element for probe attachment");
            gst_object_unref(pipeline_);
            return;
        }
        gst_object_unref(osd_element);
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find nvdsosd_0 element in pipeline. Custom OSD will not be rendered.");
        // Continue without probe if nvdsosd is not found, but custom OSD won't work
    }

    // Hook appsink for frames: Connect the new_sample_callback to the "new-sample" signal of the appsink.
    // This allows us to retrieve processed frames.
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

    // Start the GStreamer pipeline
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    RCLCPP_INFO(this->get_logger(), "Starting GStreamer main loop...");
    // Run the GLib main loop in a separate thread to avoid blocking the ROS2 spin
    gstreamer_thread_ = std::thread([this]() { g_main_loop_run(main_loop_); });
}

// === Destructor ===
// Cleans up GStreamer resources and stops the pipeline and main loop.
ObjectDetectionNode::~ObjectDetectionNode()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down pipeline...");
    // Quit the GLib main loop
    if (main_loop_)
    {
        g_main_loop_quit(main_loop_);
    }
    // Join the GStreamer thread to ensure it finishes
    if (gstreamer_thread_.joinable())
    {
        gstreamer_thread_.join(); // Corrected: Use std::thread::join() for std::thread
    }
    // Set pipeline state to NULL and unreference it to release resources
    if (pipeline_)
    {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }
    // Unreference the main loop
    if (main_loop_)
    {
        g_main_loop_unref(main_loop_);
    }
}

// === Method to cycle through detected targets ===
void ObjectDetectionNode::cycle_selected_target()
{
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);

    if (current_tracked_objects_.empty())
    {
        selected_object_id_ = UNTRACKED_OBJECT_ID; // No objects to select
        RCLCPP_INFO(this->get_logger(), "No objects to select.");
        return;
    }

    // Get all current object IDs
    std::vector<guint64> object_ids;
    for (const auto& pair : current_tracked_objects_)
    {
        object_ids.push_back(pair.first);
    }

    // Find the currently selected object's position in the vector
    auto it = std::find(object_ids.begin(), object_ids.end(), selected_object_id_);

    if (it == object_ids.end() || selected_object_id_ == UNTRACKED_OBJECT_ID)
    {
        // If no object is selected or current selection is no longer valid, select the first one
        selected_object_id_ = object_ids[0];
    }
    else
    {
        // Move to the next object, or wrap around to the first if at the end
        ++it;
        if (it == object_ids.end())
        {
            selected_object_id_ = object_ids[0];
        }
        else
        {
            selected_object_id_ = *it;
        }
    }
    RCLCPP_INFO(this->get_logger(), "Selected object ID: %lu", selected_object_id_);
}

// === Main entrypoint ===
// Standard ROS2 main function to initialize and spin the node.
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<ObjectDetectionNode>(options);

     //Example: To test cycling targets, you could set up a timer or a ROS2 service/topic
     //to call node->cycle_selected_target()
     //For manual testing, you could add a simple timer:
     rclcpp::TimerBase::SharedPtr timer = node->create_wall_timer(
         std::chrono::seconds(5),
         std::bind(&ObjectDetectionNode::cycle_selected_target, node.get()));

    rclcpp::spin(node); // Spin the ROS2 node to process callbacks
    rclcpp::shutdown();
    return 0;
}
