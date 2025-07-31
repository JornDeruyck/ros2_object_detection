// src/osd_renderer.cpp
#include "ros2_object_detection/osd_renderer.hpp"

// Standard C++ includes
#include <cmath>       // For std::abs
#include <iomanip>     // For std::setprecision
#include <sstream>     // For std::stringstream

// DeepStream metadata headers (needed for NvDsDisplayMeta functions)
#include "nvdsmeta.h"

// --- Constructor ---
// Updated constructor to accept a raw pointer to the ROS 2 node
OSDRenderer::OSDRenderer(rclcpp::Node* node_ptr) // Changed to raw pointer
    : node_(node_ptr), // Initialize the node_ member with the raw pointer
      last_fps_update_time_(std::chrono::steady_clock::now()),
      frame_count_(0),
      current_fps_(0.0)
{
    // Initialize OSD colors.
    red_color_ = {1.0, 0.0, 0.0, 1.0}; // RGBA
    green_color_ = {0.0, 1.0, 0.0, 1.0};
    blue_color_ = {0.0, 0.0, 1.0, 1.0};
    white_color_ = {1.0, 1.0, 1.0, 1.0};
    yellow_color_ = {1.0, 1.0, 0.0, 1.0};
    cyan_color_ = {0.0, 1.0, 1.0, 1.0};
    magenta_color_ = {1.0, 0.0, 1.0, 1.0};

    if (node_) {
        RCLCPP_INFO(node_->get_logger(), "OSDRenderer initialized with ROS 2 node context.");
    } else {
        RCLCPP_WARN(rclcpp::get_logger("OSDRenderer"), "OSDRenderer initialized without a valid ROS 2 node context.");
    }
}

// --- Destructor ---
OSDRenderer::~OSDRenderer()
{
    if (node_) {
        RCLCPP_INFO(node_->get_logger(), "OSDRenderer destroyed.");
    }
    // No need to delete node_ptr as it's a raw pointer owned by ObjectDetectionNode
}

// --- Public Methods ---

void OSDRenderer::update_and_display_fps(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta)
{
    std::lock_guard<std::mutex> fps_lock(fps_mutex_);

    frame_count_++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = now - last_fps_update_time_;

    if (elapsed_seconds.count() >= 1.0 || frame_count_ >= 30)
    {
        current_fps_ = frame_count_ / elapsed_seconds.count();
        frame_count_ = 0;
        last_fps_update_time_ = now;
    }

    NvDsDisplayMeta *fps_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (fps_display_meta)
    {
        fps_display_meta->num_labels = 1;
        set_text_params(
            &fps_display_meta->text_params[0],
            10, 10,
            white_color_,
            g_strdup_printf("FPS: %.2f", current_fps_)
        );
        nvds_add_display_meta_to_frame(frame_meta, fps_display_meta);
    } else {
        if (node_) RCLCPP_WARN(node_->get_logger(), "Failed to acquire display meta for FPS.");
    }
}

void OSDRenderer::render_non_selected_object_osd(NvDsObjectMeta *obj_meta)
{
    std::stringstream ss_label_conf;
    ss_label_conf << std::fixed << std::setprecision(2);

    ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id
                  << " Conf: " << obj_meta->confidence
                  << " TrkConf: " << obj_meta->tracker_confidence;

    obj_meta->rect_params.border_color = blue_color_;
    obj_meta->rect_params.border_width = 3;

    set_text_params(
        &obj_meta->text_params,
        (guint)obj_meta->rect_params.left,
        (gint)(obj_meta->rect_params.top - 25) < 0 ? (guint)(obj_meta->rect_params.top + obj_meta->rect_params.height + 5) : (guint)(obj_meta->rect_params.top - 25),
        blue_color_,
        ss_label_conf.str()
    );
}

void OSDRenderer::render_selected_object_osd(
    NvDsBatchMeta *batch_meta,
    NvDsFrameMeta *frame_meta,
    guint64 selected_object_id,
    bool selected_object_kf_initialized,
    const NvOSD_RectParams &selected_object_last_bbox,
    unsigned int selected_object_lost_frames,
    TRACKER_STATE selected_object_tracker_state,
    double predicted_vx, double predicted_vy,
    bool is_selected_object_currently_detected,
    NvDsObjectMeta *selected_obj_meta_ptr)
{
    if (!selected_object_kf_initialized) {
        return; // No KF, nothing to draw for selected object
    }

    NvOSD_ColorParams selected_color = red_color_;

    if (is_selected_object_currently_detected) {
        if (selected_obj_meta_ptr) {
            selected_obj_meta_ptr->rect_params.border_color = selected_color;
            selected_obj_meta_ptr->rect_params.border_width = 3;
            if (selected_obj_meta_ptr->text_params.display_text) {
                g_free(selected_obj_meta_ptr->text_params.display_text);
                selected_obj_meta_ptr->text_params.display_text = nullptr;
            }
        }

        NvOSD_RectParams current_bbox = selected_obj_meta_ptr ? selected_obj_meta_ptr->rect_params : selected_object_last_bbox;

        std::stringstream ss_label;
        ss_label << std::fixed << std::setprecision(2);
        ss_label << "ID: " << selected_object_id
                 << " Conf: " << (selected_obj_meta_ptr ? selected_obj_meta_ptr->confidence : 0.0)
                 << " TrkConf: " << (selected_obj_meta_ptr ? selected_obj_meta_ptr->tracker_confidence : 0.0)
                 << " Vx: " << predicted_vx
                 << " Vy: " << predicted_vy
                 << " (DS:Detected) (KF:Tracked)";

        draw_text(batch_meta, frame_meta, ss_label.str(),
                  current_bbox.left,
                  (gint)(current_bbox.top - 25) < 0 ? (guint)(current_bbox.top + current_bbox.height + 5) : (guint)(current_bbox.top - 25),
                  {0.0, 0.0, 0.0, 0.0}, // No background color
                  selected_color,
                  12);

        gfloat center_x = current_bbox.left + current_bbox.width / 2.0;
        gfloat center_y = current_bbox.top + current_bbox.height / 2.0;
        draw_reticule(batch_meta, frame_meta, center_x, center_y, 20.0, selected_color, 2);

    } else { // Object is occluded (KF is predicting)
        draw_bounding_box(batch_meta, frame_meta, selected_object_last_bbox, selected_color, 3);

        std::stringstream kf_text_ss;
        kf_text_ss << std::fixed << std::setprecision(2);
        kf_text_ss << "ID: " << selected_object_id
                   << " Vx: " << predicted_vx
                   << " Vy: " << predicted_vy
                   << " (DS:Occluded) (KF:Pred " << selected_object_lost_frames << " fr)";

        draw_text(batch_meta, frame_meta, kf_text_ss.str(),
                  selected_object_last_bbox.left,
                  (gint)(selected_object_last_bbox.top - 25) < 0 ? (guint)(selected_object_last_bbox.top + selected_object_last_bbox.height + 5) : (guint)(selected_object_last_bbox.top - 25),
                  {0.0, 0.0, 0.0, 0.0}, // No background color
                  selected_color,
                  12);

        gfloat center_x = selected_object_last_bbox.left + selected_object_last_bbox.width / 2.0;
        gfloat center_y = selected_object_last_bbox.top + selected_object_last_bbox.height / 2.0;
        draw_reticule(batch_meta, frame_meta, center_x, center_y, 20.0, selected_color, 2);
    }
}

// --- Private Methods ---

void OSDRenderer::draw_bounding_box(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, const NvOSD_RectParams &rect_params, const NvOSD_ColorParams &color, unsigned int border_width)
{
    NvDsDisplayMeta *bbox_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (bbox_display_meta) {
        bbox_display_meta->num_rects = 1;
        bbox_display_meta->rect_params[0] = rect_params;
        bbox_display_meta->rect_params[0].border_color = color;
        bbox_display_meta->rect_params[0].border_width = border_width;
        bbox_display_meta->rect_params[0].has_bg_color = 0;
        nvds_add_display_meta_to_frame(frame_meta, bbox_display_meta);
    } else {
        if (node_) RCLCPP_WARN(node_->get_logger(), "Failed to acquire display meta for bounding box.");
    }
}

void OSDRenderer::draw_text(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, const std::string &text, double x, double y, const NvOSD_ColorParams &bg_color, const NvOSD_ColorParams &text_color, unsigned int font_size)
{
    NvDsDisplayMeta *text_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (text_display_meta) {
        text_display_meta->num_labels = 1;
        set_text_params(
            &text_display_meta->text_params[0],
            (guint)x, (guint)y,
            text_color,
            text
        );
        text_display_meta->text_params[0].set_bg_clr = (bg_color.alpha > 0.0);
        text_display_meta->text_params[0].text_bg_clr = bg_color;
        text_display_meta->text_params[0].font_params.font_size = font_size;

        nvds_add_display_meta_to_frame(frame_meta, text_display_meta);
    } else {
        if (node_) RCLCPP_WARN(node_->get_logger(), "Failed to acquire display meta for text.");
    }
}


void OSDRenderer::draw_reticule(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                                       double center_x, double center_y, double size, const NvOSD_ColorParams &color, unsigned int line_width)
{
    NvDsDisplayMeta *reticule_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (reticule_display_meta)
    {
        reticule_display_meta->num_lines = 2;

        reticule_display_meta->line_params[0].x1 = (guint)(center_x - size / 2.0);
        reticule_display_meta->line_params[0].y1 = (guint)center_y;
        reticule_display_meta->line_params[0].x2 = (guint)(center_x + size / 2.0);
        reticule_display_meta->line_params[0].y2 = (guint)center_y;
        reticule_display_meta->line_params[0].line_width = line_width;
        reticule_display_meta->line_params[0].line_color = color;

        reticule_display_meta->line_params[1].x1 = (guint)center_x;
        reticule_display_meta->line_params[1].y1 = (guint)(center_y - size / 2.0);
        reticule_display_meta->line_params[1].x2 = (guint)center_x;
        reticule_display_meta->line_params[1].y2 = (guint)(center_y + size / 2.0);
        reticule_display_meta->line_params[1].line_width = line_width;
        reticule_display_meta->line_params[1].line_color = color;

        nvds_add_display_meta_to_frame(frame_meta, reticule_display_meta);
    } else {
        if (node_) RCLCPP_WARN(node_->get_logger(), "Failed to acquire display meta for reticule.");
    }
}

void OSDRenderer::draw_velocity_arrow(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, double start_x, double start_y, double end_x, double end_y, const NvOSD_ColorParams &color, unsigned int line_width)
{
    NvDsDisplayMeta *arrow_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (arrow_display_meta) {
        arrow_display_meta->num_lines = 1;
        arrow_display_meta->line_params[0].x1 = (guint)start_x;
        arrow_display_meta->line_params[0].y1 = (guint)start_y;
        arrow_display_meta->line_params[0].x2 = (guint)end_x;
        arrow_display_meta->line_params[0].y2 = (guint)end_y;
        arrow_display_meta->line_params[0].line_width = line_width;
        arrow_display_meta->line_params[0].line_color = color;
        nvds_add_display_meta_to_frame(frame_meta, arrow_display_meta);
    } else {
        if (node_) RCLCPP_WARN(node_->get_logger(), "Failed to acquire display meta for velocity arrow.");
    }
}


void OSDRenderer::set_text_params(
    NvOSD_TextParams *text_params,
    guint x_offset, guint y_offset,
    NvOSD_ColorParams font_color,
    const std::string &display_text)
{
    if (text_params->display_text) {
        g_free(text_params->display_text);
        text_params->display_text = nullptr;
    }
    text_params->display_text = g_strdup(display_text.c_str());
    text_params->x_offset = x_offset;
    text_params->y_offset = y_offset;
    text_params->font_params.font_name = (gchar *)"Sans";
    text_params->font_params.font_size = 12;
    text_params->font_params.font_color = font_color;
    text_params->set_bg_clr = 0;
}
