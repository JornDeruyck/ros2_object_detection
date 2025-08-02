// include/ros2_object_detection/osd_renderer.hpp
#ifndef OSD_RENDERER_HPP
#define OSD_RENDERER_HPP

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <rclcpp/rclcpp.hpp>

#include "nvdsmeta.h"
#include "nvll_osd_struct.h"

// Forward declarations
typedef struct _NvDsBatchMeta NvDsBatchMeta;
typedef struct _NvDsFrameMeta NvDsFrameMeta;
typedef struct _NvDsObjectMeta NvDsObjectMeta;

// Include the OSDTrackingStatus enum from object_detection.hpp
#include "ros2_object_detection/object_detection.hpp"

enum class ReticuleStyle { NONE, POINT, CIRCLE, CROSS, CROSS_DIAGONAL };

class OSDRenderer
{
public:
    explicit OSDRenderer(rclcpp::Node* node_ptr);
    ~OSDRenderer();

    void update_and_display_fps(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta);
    
    /**
     * @brief Renders OSD for non-selected objects.
     * @param batch_meta Pointer to the batch metadata for acquiring display meta.
     * @param frame_meta Pointer to the frame metadata.
     * @param obj_meta Pointer to the object's metadata.
     */
    void render_non_selected_object_osd(NvDsBatchMeta* batch_meta, NvDsFrameMeta* frame_meta, NvDsObjectMeta *obj_meta);

    void render_selected_object_osd(
        NvDsBatchMeta *batch_meta,
        NvDsFrameMeta *frame_meta,
        gint64 selected_object_id,
        const std::string& selected_object_class_label,
        OSDTrackingStatus status,
        bool is_locked_target,
        const NvOSD_RectParams &current_bbox_params,
        unsigned int selected_object_lost_frames,
        double predicted_vx, double predicted_vy,
        double camera_fov_rad
    );

private:
    rclcpp::Node* node_;
    // FPS members
    std::chrono::steady_clock::time_point last_fps_update_time_;
    unsigned int frame_count_;
    double current_fps_;
    std::mutex fps_mutex_;
    // Color members
    NvOSD_ColorParams red_color_, green_color_, blue_color_, white_color_, yellow_color_, black_color_;
    // Drawing helpers
    void draw_bounding_box(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, const NvOSD_RectParams &rect_params, const NvOSD_ColorParams &color, unsigned int border_width);
    void draw_text(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, const std::string &text, double x, double y, const NvOSD_ColorParams &text_color);
    void draw_reticule(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, double center_x, double center_y, double size, const NvOSD_ColorParams &color, unsigned int line_width, ReticuleStyle style);
    void set_text_params(NvOSD_TextParams *text_params, guint x_offset, guint y_offset, NvOSD_ColorParams font_color, const std::string &display_text, bool with_backdrop);
};

#endif // OSD_RENDERER_HPP
