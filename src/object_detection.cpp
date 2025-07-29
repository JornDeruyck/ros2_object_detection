#include "ros2_object_detection/object_detection.hpp"

// Standard C++ includes
#include <algorithm>   // For std::find, std::sort
#include <chrono>      // For FPS calculation
#include <iomanip>     // For std::setprecision
#include <mutex>       // For FPS and object mutexes
#include <sstream>     // For std::stringstream
#include <string>      // For std::string

// GStreamer & GLib includes
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <glib.h>

// DeepStream metadata headers
#include "gstnvdsmeta.h"     // For gst_buffer_get_nvds_batch_meta
#include "nvdsmeta.h"        // For NvDsDisplayMeta and other core metadata structures
#include "nvll_osd_struct.h" // For NvOSD_TextParams, NvOSD_RectParams, NvOSD_LineParams etc.

// ROS 2 message types (already in header, but good to be explicit here too)
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

// OpenCV (already in header, but good to be explicit here too)
#include <opencv2/opencv.hpp>

// Define the threshold for how many frames an object can be lost before deselection
#define SELECTED_OBJECT_LOST_THRESHOLD 20

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

        // Efficiently copy the raw JPEG data to the message vector
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
 * 5. Customizes object labels (including ID and confidence).
 * 6. Manages the state of the `selected_object_id_` (tracking and deselecting if lost).
 * 7. Draws a reticule around the currently selected object.
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

    vision_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header.stamp = node->get_clock()->now();
    detection_array_msg.header.frame_id = "camera_frame"; // Consistent frame ID

    // --- FPS Calculation ---
    // Lock mutex to protect FPS variables during update
    std::lock_guard<std::mutex> fps_lock(node->fps_mutex_);

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
    node->current_tracked_objects_.clear(); // Clear objects from previous frame, will be repopulated

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
            // Use g_strdup_printf for formatting and memory management
            fps_display_meta->text_params[0].display_text = g_strdup_printf("FPS: %.2f", node->current_fps_display_);
            fps_display_meta->text_params[0].x_offset = 10;
            fps_display_meta->text_params[0].y_offset = 10;
            fps_display_meta->text_params[0].font_params.font_name = (gchar *)"Sans";
            fps_display_meta->text_params[0].font_params.font_size = 14;
            // White color for FPS text
            fps_display_meta->text_params[0].font_params.font_color = {1.0, 1.0, 1.0, 1.0}; // RGBA (white)
            fps_display_meta->text_params[0].set_bg_clr = 0;                               // No background box

            nvds_add_display_meta_to_frame(frame_meta, fps_display_meta);
        }

        // --- Process and Modify existing NvDsObjectMeta for OSD ---
        for (GList *l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (!obj_meta) continue; // Skip if object meta is null

            // Store object's current bounding box for selection logic
            node->current_tracked_objects_[obj_meta->object_id] = obj_meta->rect_params;

            // Populate ROS Detection2D message
            vision_msgs::msg::Detection2D detection;
            detection.header.stamp = detection_array_msg.header.stamp; // Use same timestamp as array
            detection.header.frame_id = detection_array_msg.header.frame_id;

            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = std::string(obj_meta->obj_label); // Convert gchar* to std::string
            hypothesis.hypothesis.score = obj_meta->confidence;
            detection.results.push_back(hypothesis);

            // Populate bounding box coordinates
            detection.bbox.center.position.x = obj_meta->rect_params.left + obj_meta->rect_params.width / 2.0;
            detection.bbox.center.position.y = obj_meta->rect_params.top + obj_meta->rect_params.height / 2.0;
            detection.bbox.size_x = obj_meta->rect_params.width;
            detection.bbox.size_y = obj_meta->rect_params.height;
            detection_array_msg.detections.push_back(detection);

            // --- Customize Bounding Box (obj_meta->rect_params) ---
            obj_meta->rect_params.border_width = 3;   // Thicker border for visibility
            obj_meta->rect_params.has_bg_color = 0;   // No background fill for box

            // Set colors based on selection status
            if (obj_meta->object_id == node->selected_object_id_)
            {
                // Selected object: Red border
                obj_meta->rect_params.border_color = {1.0, 0.0, 0.0, 1.0}; // RGBA (red)
            }
            else
            {
                // Other objects: Blue border
                obj_meta->rect_params.border_color = {0.0, 0.0, 1.0, 1.0}; // RGBA (blue)
            }

            // --- Customize Object Label and Confidence (obj_meta->text_params) ---
            std::stringstream ss_label_conf;
            ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id
                          << " Conf: " << std::fixed << std::setprecision(2) << obj_meta->confidence;
            std::string label_conf_str = ss_label_conf.str();

            // Free previous display_text to prevent memory leaks if it was g_strdup'd by DeepStream or prior probes
            if (obj_meta->text_params.display_text)
            {
                g_free(obj_meta->text_params.display_text);
                obj_meta->text_params.display_text = nullptr; // Set to nullptr after freeing
            }
            obj_meta->text_params.display_text = g_strdup(label_conf_str.c_str());

            // Position text "on top" of the bounding box with more clearance
            obj_meta->text_params.x_offset = (guint)obj_meta->rect_params.left;
            // Use a signed integer for calculation to correctly check for negative values
            gint preferred_y_signed = (gint)(obj_meta->rect_params.top - 25);
            if (preferred_y_signed < 0) { // If it would go off-screen upwards
                obj_meta->text_params.y_offset = (guint)(obj_meta->rect_params.top + obj_meta->rect_params.height + 5);
            } else {
                obj_meta->text_params.y_offset = (guint)preferred_y_signed;
            }

            obj_meta->text_params.font_params.font_name = (gchar *)"Sans";
            obj_meta->text_params.font_params.font_size = 12;
            obj_meta->text_params.font_params.font_color = obj_meta->rect_params.border_color; // Same color as bounding box
            obj_meta->text_params.set_bg_clr = 0;                                               // No background box
        }

        // --- Manage selected_object_id_ persistence and draw Reticule ---
        if (node->selected_object_id_ != UNTRACKED_OBJECT_ID)
        {
            auto it = node->current_tracked_objects_.find(node->selected_object_id_);
            if (it != node->current_tracked_objects_.end())
            {
                // Selected object is currently visible
                node->selected_object_lost_frames_ = 0; // Reset lost frame counter

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
            }
            else
            {
                // Selected object is NOT currently visible
                node->selected_object_lost_frames_++;
                RCLCPP_DEBUG(node->get_logger(), "Selected object ID %lu lost for %u frames.",
                             node->selected_object_id_, node->selected_object_lost_frames_);

                if (node->selected_object_lost_frames_ > SELECTED_OBJECT_LOST_THRESHOLD)
                {
                    RCLCPP_INFO(node->get_logger(), "Selected object ID %lu lost for too long (%u frames). Deselecting.",
                                node->selected_object_id_, node->selected_object_lost_frames_);
                    node->selected_object_id_ = UNTRACKED_OBJECT_ID; // Deselect
                    node->selected_object_lost_frames_ = 0;          // Reset counter
                }
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
      selected_object_lost_frames_(0),                         // Initialize lost frames counter
      button0_pressed_prev_(false),                            // Initialize button 0 state for debouncing
      button1_pressed_prev_(false)                             // Initialize button 1 state for debouncing
{
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

    // Create ROS publishers
    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", qos_profile);
    compressed_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("image_raw/compressed", qos_profile);

    // Create joystick subscriber
    // Ensure `joy_node` or similar is running and publishing to `/joy`
    joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", 10, std::bind(&ObjectDetectionNode::joy_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribing to /joy topic for joystick input.");


    // Initialize GStreamer
    gst_init(nullptr, nullptr);
    main_loop_ = g_main_loop_new(nullptr, FALSE); // Create a new GLib main loop

    GError *error = nullptr;

    // Parse the GStreamer pipeline string
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline_)
    {
        RCLCPP_FATAL(this->get_logger(), "Failed to parse GStreamer pipeline: %s", error ? error->message : "Unknown error");
        if (error) g_error_free(error);
        throw std::runtime_error("GStreamer pipeline parsing failed.");
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
            RCLCPP_WARN(this->get_logger(), "Could not get sink pad from 'nvdsosd_0' element for probe attachment. Custom OSD might not function.");
        }
        gst_object_unref(osd_element); // Always unref elements obtained by gst_bin_get_by_name
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find 'nvdsosd_0' element in pipeline. Custom OSD will not be rendered.");
    }

    // Hook appsink for frames: Connect the new_sample_callback to the "new-sample" signal.
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline_), "ros_sink");
    if (appsink)
    {
        // Ensure appsink properties are set for raw data output if not already in pipeline string
        // gst_app_sink_set_emit_signals((GstAppSink*)appsink, TRUE);
        // gst_app_sink_set_drop((GstAppSink*)appsink, TRUE);
        // gst_app_sink_set_max_buffers((GstAppSink*)appsink, 1);

        g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample_callback), this);
        gst_object_unref(appsink);
        RCLCPP_INFO(this->get_logger(), "Attached new sample callback to 'ros_sink' appsink.");
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Could not find 'ros_sink' appsink in pipeline. Compressed image publication will not function.");
    }

    // Start the GStreamer pipeline
    GstStateChangeReturn state_change_ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (state_change_ret == GST_STATE_CHANGE_FAILURE) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set GStreamer pipeline to PLAYING state.");
        gst_object_unref(pipeline_);
        pipeline_ = nullptr; // Prevent double unref in destructor
        g_main_loop_unref(main_loop_);
        main_loop_ = nullptr; // Prevent double unref in destructor
        throw std::runtime_error("GStreamer pipeline failed to start.");
    }

    RCLCPP_INFO(this->get_logger(), "GStreamer pipeline started. Running GLib main loop in a separate thread...");
    // Run the GLib main loop in a separate thread to avoid blocking the ROS 2 spin
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

    if (gstreamer_thread_.joinable())
    {
        RCLCPP_INFO(this->get_logger(), "Joining GStreamer thread...");
        gstreamer_thread_.join();
    }
    RCLCPP_INFO(this->get_logger(), "GStreamer thread joined.");

    // 2. Set pipeline state to NULL and unreference it
    if (pipeline_)
    {
        RCLCPP_INFO(this->get_logger(), "Setting GStreamer pipeline to NULL state and unreferencing...");
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr; // Clear pointer after unref
    }

    // 3. Unreference the main loop
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
 * If no object is currently selected or the selected object is lost, it selects
 * the first (or last) available object.
 *
 * @param forward If true, cycles to the next object; otherwise, cycles to the previous.
 */
void ObjectDetectionNode::cycle_selected_target(bool forward)
{
    std::lock_guard<std::mutex> lock(tracked_objects_mutex_);

    if (current_tracked_objects_.empty())
    {
        selected_object_id_ = UNTRACKED_OBJECT_ID; // No objects to select
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
        // If no object is selected, or current selection is no longer valid/tracked,
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
                selected_object_id_ = object_ids[0];
            }
            else
            {
                selected_object_id_ = *it;
            }
        } else {
            // Move to the previous object, or wrap around to the last if at the beginning
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
    // Reset lost frames counter immediately upon (re)selection
    selected_object_lost_frames_ = 0;
}

// --- Callback for joystick input ---
/**
 * @brief Processes incoming `sensor_msgs::msg::Joy` messages to handle target cycling.
 *
 * This callback implements simple debouncing for joystick buttons 0 and 1.
 * Pressing button 0 cycles the selected target forward, and button 1 cycles backward.
 *
 * @param msg The received joystick message.
 */
void ObjectDetectionNode::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    // Define joystick button indices. Adjust these based on your joystick mapping.
    const int FORWARD_BUTTON_INDEX = 0; // Example: 'A' button on Xbox controller
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

    // Update previous button states for the next callback
    button0_pressed_prev_ = current_button0_pressed;
    button1_pressed_prev_ = current_button1_pressed;
}

// --- Main entrypoint ---
/**
 * @brief Main function for the ROS 2 ObjectDetectionNode executable.
 *
 * This function initializes the ROS 2 system, creates an instance of the
 * `ObjectDetectionNode`, and starts spinning the node to process callbacks.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return An integer representing the program's exit status.
 */
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv); // Initialize ROS 2

    rclcpp::NodeOptions options;
    // Potentially add options like `--ros-args -p pipeline_config:=/path/to/config.yaml`
    // to specify the config path from the command line or launch file.

    // Create and run the ObjectDetectionNode
    std::shared_ptr<ObjectDetectionNode> node = nullptr;
    try {
        node = std::make_shared<ObjectDetectionNode>(options);
        rclcpp::spin(node); // Spin the ROS 2 node to process callbacks
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Node initialization or runtime error: %s", e.what());
        // No need to re-throw, just exit.
    } catch (...) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "An unknown error occurred during node execution.");
    }


    rclcpp::shutdown(); // Shutdown ROS 2
    return 0;
}