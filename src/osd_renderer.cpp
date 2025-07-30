// src/osd_renderer.cpp
#include "ros2_object_detection/osd_renderer.hpp"

// Standard C++ includes
#include <cmath>       // For std::abs
#include <iostream>    // For debugging, can be removed in production

// DeepStream metadata headers (needed for NvDsDisplayMeta functions)
#include "nvdsmeta.h"

// --- Constructor ---
OSDRenderer::OSDRenderer()
    : last_fps_update_time_(std::chrono::steady_clock::now()),
      frame_counter_(0),
      current_fps_display_(0.0)
{
    // Constructor initializes FPS related members
}

// --- Public Methods ---

void OSDRenderer::update_and_display_fps(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta)
{
    std::lock_guard<std::mutex> fps_lock(fps_mutex_);

    frame_counter_++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = now - last_fps_update_time_;

    if (elapsed_seconds.count() >= 1.0 || frame_counter_ >= 30)
    {
        current_fps_display_ = frame_counter_ / elapsed_seconds.count();
        frame_counter_ = 0;
        last_fps_update_time_ = now;
    }

    NvDsDisplayMeta *fps_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (fps_display_meta)
    {
        fps_display_meta->num_labels = 1;
        set_text_params(
            &fps_display_meta->text_params[0],
            10, 10,
            {1.0, 1.0, 1.0, 1.0}, // White
            g_strdup_printf("FPS: %.2f", current_fps_display_)
        );
        nvds_add_display_meta_to_frame(frame_meta, fps_display_meta);
    }
}

void OSDRenderer::render_non_selected_object_osd(NvDsObjectMeta *obj_meta)
{
    std::stringstream ss_label_conf;
    ss_label_conf << std::fixed << std::setprecision(2);

    ss_label_conf << obj_meta->obj_label << " ID: " << obj_meta->object_id
                  << " Conf: " << obj_meta->confidence
                  << " TrkConf: " << obj_meta->tracker_confidence;

    obj_meta->rect_params.border_color = {0.0, 0.0, 1.0, 1.0}; // Default blue for non-selected objects
    obj_meta->rect_params.border_width = 3;

    set_text_params(
        &obj_meta->text_params,
        (guint)obj_meta->rect_params.left,
        (gint)(obj_meta->rect_params.top - 25) < 0 ? (guint)(obj_meta->rect_params.top + obj_meta->rect_params.height + 5) : (guint)(obj_meta->rect_params.top - 25),
        {0.0, 0.0, 1.0, 1.0}, // Blue text
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

    NvOSD_ColorParams selected_color = {1.0, 0.0, 0.0, 1.0}; // Red color for selected object

    // Determine which bounding box and text to draw
    if (is_selected_object_currently_detected) {
        // Object is detected by DeepStream: Use its NvDsObjectMeta for bounding box and clear its text
        if (selected_obj_meta_ptr) {
            selected_obj_meta_ptr->rect_params.border_color = selected_color;
            selected_obj_meta_ptr->rect_params.border_width = 3;
            if (selected_obj_meta_ptr->text_params.display_text) {
                g_free(selected_obj_meta_ptr->text_params.display_text);
                selected_obj_meta_ptr->text_params.display_text = nullptr;
            }
        }

        // Draw custom text and reticule using DeepStream's detected bbox
        NvOSD_RectParams current_bbox = selected_obj_meta_ptr ? selected_obj_meta_ptr->rect_params : selected_object_last_bbox;
        
        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (display_meta) {
            display_meta->num_labels = 1;
            std::stringstream ss_label;
            ss_label << std::fixed << std::setprecision(2);
            ss_label << "ID: " << selected_object_id
                     << " Conf: " << (selected_obj_meta_ptr ? selected_obj_meta_ptr->confidence : 0.0)
                     << " TrkConf: " << (selected_obj_meta_ptr ? selected_obj_meta_ptr->tracker_confidence : 0.0)
                     << " Vx: " << predicted_vx
                     << " Vy: " << predicted_vy
                     << " (DS:Detected) (KF:Tracked)";

            set_text_params(
                &display_meta->text_params[0],
                (guint)current_bbox.left,
                (gint)(current_bbox.top - 25) < 0 ? (guint)(current_bbox.top + current_bbox.height + 5) : (guint)(current_bbox.top - 25),
                selected_color, // Red text
                ss_label.str()
            );
            nvds_add_display_meta_to_frame(frame_meta, display_meta);
        }

        // Draw reticule on top of DeepStream's box
        gfloat center_x = current_bbox.left + current_bbox.width / 2.0;
        gfloat center_y = current_bbox.top + current_bbox.height / 2.0;
        draw_reticule(batch_meta, frame_meta, center_x, center_y, 20.0, selected_color);

    } else { // Object is occluded (KF is predicting)
        // Draw KF-predicted bounding box
        NvDsDisplayMeta *bbox_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (bbox_display_meta) {
            bbox_display_meta->num_rects = 1;
            bbox_display_meta->rect_params[0] = selected_object_last_bbox;
            bbox_display_meta->rect_params[0].border_color = selected_color; // Red border
            bbox_display_meta->rect_params[0].border_width = 3;
            bbox_display_meta->rect_params[0].has_bg_color = 0;
            nvds_add_display_meta_to_frame(frame_meta, bbox_display_meta);
        }

        // Draw text with KF prediction info
        NvDsDisplayMeta *text_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        if (text_display_meta) {
            text_display_meta->num_labels = 1;
            std::stringstream kf_text_ss;
            kf_text_ss << std::fixed << std::setprecision(2);
            kf_text_ss << "ID: " << selected_object_id
                       << " Vx: " << predicted_vx
                       << " Vy: " << predicted_vy
                       << " (DS:Occluded) (KF:Pred " << selected_object_lost_frames << " fr)";

            set_text_params(
                &text_display_meta->text_params[0],
                (guint)selected_object_last_bbox.left,
                (gint)(selected_object_last_bbox.top - 25) < 0 ? (guint)(selected_object_last_bbox.top + selected_object_last_bbox.height + 5) : (guint)(selected_object_last_bbox.top - 25),
                selected_color, // Red text
                kf_text_ss.str()
            );
            nvds_add_display_meta_to_frame(frame_meta, text_display_meta);
        }

        // Draw reticule at KF-predicted center
        gfloat center_x = selected_object_last_bbox.left + selected_object_last_bbox.width / 2.0;
        gfloat center_y = selected_object_last_bbox.top + selected_object_last_bbox.height / 2.0;
        draw_reticule(batch_meta, frame_meta, center_x, center_y, 20.0, selected_color);
    }
}

// --- Private Methods ---

void OSDRenderer::draw_reticule(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                                       gfloat center_x, gfloat center_y, gfloat size, NvOSD_ColorParams color)
{
    NvDsDisplayMeta *reticule_display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (reticule_display_meta)
    {
        reticule_display_meta->num_lines = 2; // Two lines for crosshair

        // Horizontal line of the reticule
        reticule_display_meta->line_params[0].x1 = (guint)(center_x - size / 2.0);
        reticule_display_meta->line_params[0].y1 = (guint)center_y;
        reticule_display_meta->line_params[0].x2 = (guint)(center_x + size / 2.0);
        reticule_display_meta->line_params[0].y2 = (guint)center_y;
        reticule_display_meta->line_params[0].line_width = 2;
        reticule_display_meta->line_params[0].line_color = color;

        // Vertical line of the reticule
        reticule_display_meta->line_params[1].x1 = (guint)center_x;
        reticule_display_meta->line_params[1].y1 = (guint)(center_y - size / 2.0);
        reticule_display_meta->line_params[1].x2 = (guint)center_x;
        reticule_display_meta->line_params[1].y2 = (guint)(center_y + size / 2.0);
        reticule_display_meta->line_params[1].line_width = 2;
        reticule_display_meta->line_params[1].line_color = color;

        nvds_add_display_meta_to_frame(frame_meta, reticule_display_meta);
    } else {
        std::cerr << "OSDRenderer: Failed to acquire display meta for reticule." << std::endl;
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
    text_params->set_bg_clr = 0; // No background box
}
