// src/osd_renderer.cpp
#include "ros2_object_detection/osd_renderer.hpp"

// Standard C++ includes
#include <cmath>       // For std::abs, std::sqrt, std::tan
#include <iomanip>     // For std::setprecision
#include <sstream>     // For std::stringstream

// DeepStream metadata headers (needed for NvDsDisplayMeta functions)
#include "nvdsmeta.h"

// --- Constructor ---
OSDRenderer::OSDRenderer(rclcpp::Node* node_ptr)
    : node_(node_ptr),
      last_fps_update_time_(std::chrono::steady_clock::now()),
      frame_count_(0),
      current_fps_(0.0)
{
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
                  << " Conf: " << obj_meta->confidence;
    // Removed TrkConf as it's not always reliable and adds length.

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
    const std::string& selected_object_class_label,
    TrackingStatus status,
    bool kf_initialized,
    const NvOSD_RectParams &current_bbox_params,
    unsigned int selected_object_lost_frames,
    double predicted_vx, double predicted_vy,
    double center_x, double center_y,
    double camera_fov_rad
)
{
    if (!kf_initialized) {
        return; // No KF, nothing to draw for selected object
    }

    NvOSD_ColorParams selected_color = red_color_;
    const guint frame_width = frame_meta->source_frame_width;
    const guint frame_height = frame_meta->source_frame_height;

    // Calculate vertical FOV based on aspect ratio
    const double aspect_ratio = (double)frame_width / (double)frame_height;
    const double camera_fov_v_rad = 2.0 * std::atan(std::tan(camera_fov_rad / 2.0) / aspect_ratio);

    // Convert pixel position to radians from center
    double pos_x_rad = ((center_x - frame_width / 2.0) / frame_width) * camera_fov_rad;
    double pos_y_rad = ((center_y - frame_height / 2.0) / frame_height) * camera_fov_v_rad;

    // Convert pixel velocity to radians/second
    double vx_rad_s = 0.0;
    double vy_rad_s = 0.0;
    if (current_fps_ > 0.0) {
        vx_rad_s = (predicted_vx / frame_width) * camera_fov_rad * current_fps_;
        vy_rad_s = (predicted_vy / frame_height) * camera_fov_v_rad * current_fps_;
    }

    // Draw the bounding box and text
    draw_bounding_box(batch_meta, frame_meta, current_bbox_params, selected_color, 3);
    
    // Determine the status string
    std::string status_string;
    switch (status) {
        case DETECTED:
            status_string = "STATUS: DETECTED";
            break;
        case OCCLUDED:
            status_string = "STATUS: OCCLUDED (Lost for " + std::to_string(selected_object_lost_frames) + " frames)";
            break;
        case TRACKED:
            status_string = "STATUS: TRACKED";
            break;
    }

    // Line 1: ID, Class, Velocity
    std::stringstream ss_line1;
    ss_line1 << std::fixed << std::setprecision(3);
    ss_line1 << "ID: " << selected_object_id
             << " Class: " << selected_object_class_label
             << " Vx: " << vx_rad_s << " rad/s"
             << " Vy: " << vy_rad_s << " rad/s";

    // Line 2: Status and Position
    std::stringstream ss_line2;
    ss_line2 << std::fixed << std::setprecision(3);
    ss_line2 << status_string
             << " Pos: (" << pos_x_rad << " rad, " << pos_y_rad << " rad)";

    // Calculate text position
    gint text_y_offset_top = (gint)(current_bbox_params.top - 25) < 0 ? (guint)(current_bbox_params.top + current_bbox_params.height + 5) : (guint)(current_bbox_params.top - 25);
    gint text_y_offset_bottom = text_y_offset_top + 20;

    // Draw first line of text
    draw_text(batch_meta, frame_meta, ss_line1.str(),
              current_bbox_params.left,
              text_y_offset_top,
              {0.0, 0.0, 0.0, 0.0},
              selected_color,
              12);

    // Draw second line of text (below the first)
    draw_text(batch_meta, frame_meta, ss_line2.str(),
              current_bbox_params.left,
              text_y_offset_bottom,
              {0.0, 0.0, 0.0, 0.0},
              selected_color,
              12);

    // Draw reticule at the center of the bounding box
    draw_reticule(batch_meta, frame_meta, center_x, center_y, 50.0, selected_color, 1, ReticuleStyle::CROSS_DIAGONAL);
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

void OSDRenderer::draw_text(
    NvDsBatchMeta *batch_meta,
    NvDsFrameMeta *frame_meta,
    const std::string &text,
    double x,
    double y,
    const NvOSD_ColorParams &bg_color,
    const NvOSD_ColorParams &text_color,
    unsigned int font_size)
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
                                       double center_x, double center_y, double size, const NvOSD_ColorParams &color, unsigned int line_width, ReticuleStyle style)
{
    if (style == ReticuleStyle::NONE) {
        return; // Do not draw anything
    }

    NvDsDisplayMeta *reticule_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (!reticule_display_meta) {
        if (node_) RCLCPP_WARN(node_->get_logger(), "Failed to acquire display meta for reticule.");
        return;
    }

    reticule_display_meta->num_lines = 0; // Reset line count for each style
    reticule_display_meta->num_circles = 0; // Reset circle count for each style

    switch (style) {
        case ReticuleStyle::POINT: {
            reticule_display_meta->num_circles = 1;
            reticule_display_meta->circle_params[0].xc = (guint)center_x;
            reticule_display_meta->circle_params[0].yc = (guint)center_y;
            reticule_display_meta->circle_params[0].radius = line_width; // Use line_width for point size
            reticule_display_meta->circle_params[0].circle_color = color;
            reticule_display_meta->circle_params[0].has_bg_color = 1; // Filled circle
            reticule_display_meta->circle_params[0].bg_color = color;
            break;
        }
        case ReticuleStyle::CIRCLE: {
            reticule_display_meta->num_circles = 1;
            reticule_display_meta->circle_params[0].xc = (guint)center_x;
            reticule_display_meta->circle_params[0].yc = (guint)center_y;
            reticule_display_meta->circle_params[0].radius = size / 2.0; // Radius based on size
            reticule_display_meta->circle_params[0].circle_color = color;
            reticule_display_meta->circle_params[0].circle_width = line_width;
            reticule_display_meta->circle_params[0].has_bg_color = 0; // Outline only
            break;
        }
        case ReticuleStyle::CROSS: {
            reticule_display_meta->num_lines = 2;
            // Horizontal line
            reticule_display_meta->line_params[0].x1 = (guint)(center_x - size / 2.0);
            reticule_display_meta->line_params[0].y1 = (guint)center_y;
            reticule_display_meta->line_params[0].x2 = (guint)(center_x + size / 2.0);
            reticule_display_meta->line_params[0].y2 = (guint)center_y;
            reticule_display_meta->line_params[0].line_width = line_width;
            reticule_display_meta->line_params[0].line_color = color;

            // Vertical line
            reticule_display_meta->line_params[1].x1 = (guint)center_x;
            reticule_display_meta->line_params[1].y1 = (guint)(center_y - size / 2.0);
            reticule_display_meta->line_params[1].x2 = (guint)center_x;
            reticule_display_meta->line_params[1].y2 = (guint)(center_y + size / 2.0);
            reticule_display_meta->line_params[1].line_width = line_width;
            reticule_display_meta->line_params[1].line_color = color;
            break;
        }
        case ReticuleStyle::CROSS_DIAGONAL: {
            // Calculate a gap size for the lines from the center dot
            const double gap_from_center = line_width * 5.0; // Adjust this value as needed for the desired gap
            const double half_side = size / 2.0;

            // Calculate start and end points for the diagonal lines
            const double end_x_offset = half_side;
            const double end_y_offset = half_side;
            
            // Check if the lines would be long enough to draw
            if (gap_from_center < half_side) {
                // We'll draw 4 lines for the cross
                reticule_display_meta->num_lines = 4;
                reticule_display_meta->num_circles = 1; // for the central dot

                // Central dot
                reticule_display_meta->circle_params[0].xc = (guint)center_x;
                reticule_display_meta->circle_params[0].yc = (guint)center_y;
                reticule_display_meta->circle_params[0].radius = line_width; // Size of the dot
                reticule_display_meta->circle_params[0].circle_color = color;
                reticule_display_meta->circle_params[0].has_bg_color = 1; // Filled circle
                reticule_display_meta->circle_params[0].bg_color = color;

                // Diagonal 1 (top-left to bottom-right)
                reticule_display_meta->line_params[0].x1 = (guint)(center_x - end_x_offset);
                reticule_display_meta->line_params[0].y1 = (guint)(center_y - end_y_offset);
                reticule_display_meta->line_params[0].x2 = (guint)(center_x - gap_from_center);
                reticule_display_meta->line_params[0].y2 = (guint)(center_y - gap_from_center);
                reticule_display_meta->line_params[0].line_width = line_width;
                reticule_display_meta->line_params[0].line_color = color;

                reticule_display_meta->line_params[1].x1 = (guint)(center_x + gap_from_center);
                reticule_display_meta->line_params[1].y1 = (guint)(center_y + gap_from_center);
                reticule_display_meta->line_params[1].x2 = (guint)(center_x + end_x_offset);
                reticule_display_meta->line_params[1].y2 = (guint)(center_y + end_y_offset);
                reticule_display_meta->line_params[1].line_width = line_width;
                reticule_display_meta->line_params[1].line_color = color;

                // Diagonal 2 (bottom-left to top-right)
                reticule_display_meta->line_params[2].x1 = (guint)(center_x - end_x_offset);
                reticule_display_meta->line_params[2].y1 = (guint)(center_y + end_y_offset);
                reticule_display_meta->line_params[2].x2 = (guint)(center_x - gap_from_center);
                reticule_display_meta->line_params[2].y2 = (guint)(center_y + gap_from_center);
                reticule_display_meta->line_params[2].line_width = line_width;
                reticule_display_meta->line_params[2].line_color = color;

                reticule_display_meta->line_params[3].x1 = (guint)(center_x + gap_from_center);
                reticule_display_meta->line_params[3].y1 = (guint)(center_y - gap_from_center);
                reticule_display_meta->line_params[3].x2 = (guint)(center_x + end_x_offset);
                reticule_display_meta->line_params[3].y2 = (guint)(center_y - end_y_offset);
                reticule_display_meta->line_params[3].line_width = line_width;
                reticule_display_meta->line_params[3].line_color = color;
            } else {
                reticule_display_meta->num_lines = 0;
            }
            break;
        }
        case ReticuleStyle::NONE:
        default:
            break;
    }

    if (reticule_display_meta->num_lines > 0 || reticule_display_meta->num_circles > 0) {
        nvds_add_display_meta_to_frame(frame_meta, reticule_display_meta);
    } else {
        // Since no display meta was added, we don't need to do anything.
        // It will be cleaned up automatically as it was not added to a frame.
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