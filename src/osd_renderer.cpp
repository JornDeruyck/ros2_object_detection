// src/osd_renderer.cpp
#include "ros2_object_detection/osd_renderer.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>
#include <algorithm> // For std::max

// Helper to format numbers for stable OSD text
std::string format_fixed_width(double value, int width, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::showpos << std::setprecision(precision) << std::setw(width) << value;
    return ss.str();
}

OSDRenderer::OSDRenderer(rclcpp::Node* node_ptr)
    : node_(node_ptr),
      last_fps_update_time_(std::chrono::steady_clock::now()),
      frame_count_(0),
      current_fps_(0.0)
{
    red_color_ = {1.0, 0.0, 0.0, 1.0};
    green_color_ = {0.0, 1.0, 0.0, 1.0};
    blue_color_ = {0.0, 0.0, 1.0, 1.0};
    white_color_ = {1.0, 1.0, 1.0, 1.0};
    yellow_color_ = {1.0, 1.0, 0.0, 1.0};
    black_color_ = {0.0, 0.0, 0.0, 0.7}; // Semi-transparent black for backdrops
}

OSDRenderer::~OSDRenderer() {}

void OSDRenderer::update_and_display_fps(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta)
{
    std::lock_guard<std::mutex> fps_lock(fps_mutex_);
    frame_count_++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = now - last_fps_update_time_;
    if (elapsed_seconds.count() >= 1.0 || frame_count_ >= 30) {
        current_fps_ = frame_count_ / elapsed_seconds.count();
        frame_count_ = 0;
        last_fps_update_time_ = now;
    }
    
    std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps_));
    draw_text(batch_meta, frame_meta, fps_text, 10, 10, white_color_);
}

void OSDRenderer::render_non_selected_object_osd(NvDsBatchMeta* /*batch_meta*/, NvDsFrameMeta* /*frame_meta*/, NvDsObjectMeta *obj_meta)
{
    std::stringstream ss_label_conf;
    ss_label_conf << obj_meta->obj_label << " ID:" << obj_meta->object_id;

    obj_meta->rect_params.border_width = 2;
    obj_meta->rect_params.border_color = blue_color_;
    obj_meta->rect_params.has_bg_color = 0;

    // --- Centering Logic for Non-Selected Objects ---
    const int avg_char_width = 8;
    const double bbox_center_x = obj_meta->rect_params.left + obj_meta->rect_params.width / 2.0;
    double text_x_pos = bbox_center_x - (ss_label_conf.str().length() * avg_char_width / 2.0);
    text_x_pos = std::max(0.0, text_x_pos);

    gint text_y_pos = (gint)(obj_meta->rect_params.top - 25) < 0 
        ? (guint)(obj_meta->rect_params.top + obj_meta->rect_params.height + 5) 
        : (guint)(obj_meta->rect_params.top - 25);

    set_text_params(
        &obj_meta->text_params,
        (guint)text_x_pos,
        text_y_pos,
        blue_color_,
        ss_label_conf.str(),
        true // with backdrop
    );
}

void OSDRenderer::render_selected_object_osd(
    NvDsBatchMeta *batch_meta,
    NvDsFrameMeta *frame_meta,
    guint64 selected_object_id,
    const std::string& selected_object_class_label,
    OSDTrackingStatus status,
    bool is_locked_target,
    const NvOSD_RectParams &current_bbox_params,
    unsigned int selected_object_lost_frames,
    double predicted_vx, double predicted_vy,
    double camera_fov_rad
)
{
    NvOSD_ColorParams selected_color = is_locked_target ? green_color_ : red_color_;
    if (status == OSDTrackingStatus::OCCLUDED) {
        selected_color = yellow_color_;
    }
    
    double center_x = current_bbox_params.left + current_bbox_params.width / 2.0;
    double center_y = current_bbox_params.top + current_bbox_params.height / 2.0;

    draw_bounding_box(batch_meta, frame_meta, current_bbox_params, selected_color, 1);
    draw_reticule(batch_meta, frame_meta, center_x, center_y, 50.0, selected_color, 2, ReticuleStyle::CROSS_DIAGONAL);

    const double aspect_ratio = (double)frame_meta->source_frame_width / frame_meta->source_frame_height;
    const double camera_fov_v_rad = 2.0 * std::atan(std::tan(camera_fov_rad / 2.0) / aspect_ratio);
    double pos_x_rad = ((center_x - frame_meta->source_frame_width / 2.0) / frame_meta->source_frame_width) * camera_fov_rad;
    double pos_y_rad = ((center_y - frame_meta->source_frame_height / 2.0) / frame_meta->source_frame_height) * camera_fov_v_rad;
    double vx_rad_s = (current_fps_ > 0.0) ? (predicted_vx / frame_meta->source_frame_width) * camera_fov_rad * current_fps_ : 0.0;
    double vy_rad_s = (current_fps_ > 0.0) ? (predicted_vy / frame_meta->source_frame_height) * camera_fov_v_rad * current_fps_ : 0.0;

    std::string status_string;
    if (is_locked_target) {
        status_string = "LOCKED";
    } else {
        status_string = (status == OSDTrackingStatus::DETECTED) 
            ? "DETECTED" 
            : "OCCLUDED (" + std::to_string(selected_object_lost_frames) + "f)";
    }

    std::stringstream ss_line1, ss_line2;
    ss_line1 << "ID: " << selected_object_id << " " << selected_object_class_label << " | " << status_string;
    ss_line2 << "P:(" << format_fixed_width(pos_x_rad, 6, 3) << "," << format_fixed_width(pos_y_rad, 6, 3) << ") rad"
             << "V:(" << format_fixed_width(vx_rad_s, 6, 3) << "," << format_fixed_width(vy_rad_s, 6, 3) << ") rad/s";

    const double bbox_center_x = current_bbox_params.left + current_bbox_params.width / 2.0;
    const int avg_char_width = 8;

    double x_line1 = bbox_center_x - (ss_line1.str().length() * avg_char_width / 2.0);
    double x_line2 = bbox_center_x - (ss_line2.str().length() * avg_char_width / 2.0);
    x_line1 = std::max(0.0, x_line1);
    x_line2 = std::max(0.0, x_line2);

    // --- Repositioned Text Logic ---
    gint y_line1, y_line2;
    if ((gint)current_bbox_params.top > 30) { // Enough space for one line above
        y_line1 = current_bbox_params.top - 30;
        y_line2 = current_bbox_params.top + 0; // Position second line just inside the box
    } else { // Not enough space, place both below
        y_line1 = current_bbox_params.top + current_bbox_params.height + 5;
        y_line2 = y_line1 + 25;
    }

    draw_text(batch_meta, frame_meta, ss_line1.str(), x_line1, y_line1, selected_color);
    draw_text(batch_meta, frame_meta, ss_line2.str(), x_line2, y_line2, selected_color);
}

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
    }
}

void OSDRenderer::draw_text(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, const std::string &text, double x, double y, const NvOSD_ColorParams &text_color)
{
    NvDsDisplayMeta *text_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (text_display_meta) {
        text_display_meta->num_labels = 1;
        set_text_params(&text_display_meta->text_params[0], (guint)x, (guint)y, text_color, text, true);
        nvds_add_display_meta_to_frame(frame_meta, text_display_meta);
    }
}

void OSDRenderer::draw_reticule(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, double center_x, double center_y, double size, const NvOSD_ColorParams &color, unsigned int line_width, ReticuleStyle style)
{
    if (style == ReticuleStyle::NONE) return;
    NvDsDisplayMeta *reticule_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (!reticule_display_meta) return;
    reticule_display_meta->num_lines = 0;
    reticule_display_meta->num_circles = 0;
    switch (style) {
        case ReticuleStyle::POINT:
            reticule_display_meta->num_circles = 1;
            reticule_display_meta->circle_params[0] = { (guint)center_x, (guint)center_y, line_width, color, 1, color, 0 };
            break;
        case ReticuleStyle::CIRCLE:
            reticule_display_meta->num_circles = 1;
            reticule_display_meta->circle_params[0] = { (guint)center_x, (guint)center_y, (guint)(size / 2.0), color, 0, {}, line_width };
            break;
        case ReticuleStyle::CROSS:
            reticule_display_meta->num_lines = 2;
            reticule_display_meta->line_params[0] = { (guint)(center_x - size / 2.0), (guint)center_y, (guint)(center_x + size / 2.0), (guint)center_y, line_width, color };
            reticule_display_meta->line_params[1] = { (guint)center_x, (guint)(center_y - size / 2.0), (guint)center_x, (guint)(center_y + size / 2.0), line_width, color };
            break;
        case ReticuleStyle::CROSS_DIAGONAL: {
            const double gap = line_width * 5.0;
            const double half_side = size / 2.0;
            if (gap < half_side) {
                reticule_display_meta->num_lines = 4;
                reticule_display_meta->num_circles = 1;
                reticule_display_meta->circle_params[0] = { (guint)center_x, (guint)center_y, line_width, color, 1, color, 0 };
                reticule_display_meta->line_params[0] = { (guint)(center_x - half_side), (guint)(center_y - half_side), (guint)(center_x - gap), (guint)(center_y - gap), line_width, color };
                reticule_display_meta->line_params[1] = { (guint)(center_x + gap), (guint)(center_y + gap), (guint)(center_x + half_side), (guint)(center_y + half_side), line_width, color };
                reticule_display_meta->line_params[2] = { (guint)(center_x - half_side), (guint)(center_y + half_side), (guint)(center_x - gap), (guint)(center_y + gap), line_width, color };
                reticule_display_meta->line_params[3] = { (guint)(center_x + gap), (guint)(center_y - gap), (guint)(center_x + half_side), (guint)(center_y - half_side), line_width, color };
            }
            break;
        }
        default: break;
    }
    if (reticule_display_meta->num_lines > 0 || reticule_display_meta->num_circles > 0) {
        nvds_add_display_meta_to_frame(frame_meta, reticule_display_meta);
    }
}

void OSDRenderer::set_text_params(NvOSD_TextParams *text_params, guint x_offset, guint y_offset, NvOSD_ColorParams font_color, const std::string &display_text, bool with_backdrop)
{
    if (text_params->display_text) {
        g_free(text_params->display_text);
    }
    text_params->display_text = g_strdup(display_text.c_str());
    text_params->x_offset = x_offset;
    text_params->y_offset = y_offset;
    text_params->font_params.font_name = (gchar *)"Sans";
    text_params->font_params.font_size = 14;
    text_params->font_params.font_color = font_color;
    if (with_backdrop) {
        text_params->set_bg_clr = 1;
        text_params->text_bg_clr = black_color_;
    } else {
        text_params->set_bg_clr = 0;
    }
}
