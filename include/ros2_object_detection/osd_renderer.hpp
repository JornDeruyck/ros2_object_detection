// include/ros2_object_detection/osd_renderer.hpp
#ifndef OSD_RENDERER_HPP
#define OSD_RENDERER_HPP

#include <memory> // For std::unique_ptr (though not directly used here, good practice)
#include <string> // For std::string
#include <vector> // For std::vector
#include <chrono> // For std::chrono::steady_clock
#include <mutex>  // For std::mutex
#include <rclcpp/rclcpp.hpp> // For rclcpp::Node (raw pointer)

// DeepStream metadata headers
#include "nvdsmeta.h"
#include "nvll_osd_struct.h"
#include "nvds_tracker_meta.h" // For TRACKER_STATE enum

// Forward declarations for GStreamer types
typedef struct _NvDsBatchMeta NvDsBatchMeta;
typedef struct _NvDsFrameMeta NvDsFrameMeta;
typedef struct _NvDsObjectMeta NvDsObjectMeta;

/**
 * @brief Enumeration for different reticule styles.
 */
enum class ReticuleStyle
{
    NONE,           ///< No reticule drawn.
    POINT,          ///< A small central point.
    CIRCLE,         ///< A circle.
    CROSS,          ///< A simple cross (horizontal and vertical lines).
    CROSS_DIAGONAL  ///< A cross with additional diagonal lines.
    // Add more styles here if desired
};

/**
 * @class OSDRenderer
 * @brief Manages custom On-Screen Display (OSD) rendering for DeepStream frames.
 *
 * This class handles drawing FPS, selected object highlights, reticules, and velocity
 * information directly onto the DeepStream video frames using NvOSD functions.
 */
class OSDRenderer
{
public:
    /**
     * @brief Constructor for the OSDRenderer.
     * @param node_ptr A raw pointer to the ROS 2 node, allowing access to its logger, clock, etc.
     */
    explicit OSDRenderer(rclcpp::Node* node_ptr);

    /**
     * @brief Destructor for the OSDRenderer.
     */
    ~OSDRenderer();

    /**
     * @brief Updates FPS calculation and adds FPS text to the frame's OSD.
     * @param batch_meta Pointer to the NvDsBatchMeta.
     * @param frame_meta Pointer to the NvDsFrameMeta of the current frame.
     */
    void update_and_display_fps(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta);

    /**
     * @brief Renders OSD for non-selected objects.
     * This typically involves drawing their bounding boxes and labels as provided by DeepStream.
     * @param obj_meta Pointer to the NvDsObjectMeta of the non-selected object.
     */
    void render_non_selected_object_osd(NvDsObjectMeta *obj_meta);

    /**
     * @brief Renders OSD for the currently selected object, including highlight, reticule, and velocity.
     * @param batch_meta Pointer to the NvDsBatchMeta.
     * @param frame_meta Pointer to the NvDsFrameMeta of the current frame.
     * @param selected_object_id The ID of the currently selected object.
     * @param kf_initialized True if the Kalman Filter for the selected object is initialized.
     * @param selected_object_last_bbox The last known bounding box (detected or predicted) of the selected object.
     * @param selected_object_lost_frames Number of frames the selected object has been lost by the detector.
     * @param selected_object_tracker_state DeepStream tracker's reported state for the selected object.
     * @param predicted_vx Predicted X velocity from Kalman Filter.
     * @param predicted_vy Predicted Y velocity from Kalman Filter.
     * @param selected_object_found_in_frame True if the selected object was detected in the current frame.
     * @param current_selected_obj_meta_ptr Pointer to the NvDsObjectMeta of the selected object if found in the current frame.
     */
    void render_selected_object_osd(
        NvDsBatchMeta *batch_meta,
        NvDsFrameMeta *frame_meta,
        guint64 selected_object_id,
        bool kf_initialized,
        const NvOSD_RectParams &selected_object_last_bbox,
        unsigned int selected_object_lost_frames,
        TRACKER_STATE selected_object_tracker_state,
        double predicted_vx,
        double predicted_vy,
        bool selected_object_found_in_frame,
        NvDsObjectMeta *current_selected_obj_meta_ptr
    );

private:
    rclcpp::Node* node_; ///< Raw pointer to the ROS 2 node for logging/clock access.

    // FPS calculation members
    std::chrono::steady_clock::time_point last_fps_update_time_; ///< Last time FPS was updated.
    unsigned int frame_count_;                                   ///< Number of frames processed since last FPS update.
    double current_fps_;                                         ///< Current calculated FPS.
    std::mutex fps_mutex_;                                       ///< Mutex to protect FPS variables from concurrent access.


    // OSD parameters (colors, line widths, etc.)
    NvOSD_ColorParams red_color_;
    NvOSD_ColorParams green_color_;
    NvOSD_ColorParams blue_color_;
    NvOSD_ColorParams white_color_;
    NvOSD_ColorParams yellow_color_;
    NvOSD_ColorParams cyan_color_;
    NvOSD_ColorParams magenta_color_;

    // Helper functions for OSD drawing
    void draw_bounding_box(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, const NvOSD_RectParams &rect_params, const NvOSD_ColorParams &color, unsigned int border_width);
    void draw_text(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, const std::string &text, double x, double y, const NvOSD_ColorParams &bg_color, const NvOSD_ColorParams &text_color, unsigned int font_size);
    // Updated draw_reticule signature
    void draw_reticule(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, double center_x, double center_y, double size, const NvOSD_ColorParams &color, unsigned int line_width, ReticuleStyle style);
    void draw_velocity_arrow(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, double start_x, double start_y, double end_x, double end_y, const NvOSD_ColorParams &color, unsigned int line_width);

    // Helper to set common text parameters for OSD labels.
    void set_text_params(
        NvOSD_TextParams *text_params,
        guint x_offset, guint y_offset,
        NvOSD_ColorParams font_color,
        const std::string &display_text);
};

#endif // OSD_RENDERER_HPP
