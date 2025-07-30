// include/ros2_object_detection/osd_renderer.hpp
#ifndef OSD_RENDERER_HPP
#define OSD_RENDERER_HPP

// Standard C++ includes
#include <chrono>      // For std::chrono::steady_clock
#include <iomanip>     // For std::setprecision
#include <mutex>       // For std::mutex
#include <sstream>     // For std::stringstream
#include <string>      // For std::string

// DeepStream metadata headers
#include "nvdsmeta.h"        // For NvDsDisplayMeta and other core metadata structures
#include "nvll_osd_struct.h" // For NvOSD_TextParams, NvOSD_RectParams, NvOSD_LineParams etc.
#include "nvds_tracker_meta.h" // For TRACKER_STATE enum

// Custom local includes
#include "ros2_object_detection/constants.hpp" // For UNTRACKED_OBJECT_ID, KF_LOST_THRESHOLD

// Forward declarations for GStreamer types.
typedef struct _NvDsFrameMeta NvDsFrameMeta;
typedef struct _NvDsBatchMeta NvDsBatchMeta;
typedef struct _NvDsObjectMeta NvDsObjectMeta;

/**
 * @class OSDRenderer
 * @brief Manages On-Screen Display (OSD) elements for DeepStream output.
 *
 * This class handles drawing FPS, bounding boxes, labels, and a reticule for a selected object.
 * It aims to provide a unified look and feel for OSD elements, regardless of whether
 * an object is selected or occluded.
 */
class OSDRenderer
{
public:
    /**
     * @brief Constructor for OSDRenderer.
     */
    OSDRenderer();

    /**
     * @brief Updates and displays the Frames Per Second (FPS) on the OSD.
     * @param batch_meta Pointer to the current NvDsBatchMeta.
     * @param frame_meta Pointer to the current NvDsFrameMeta.
     */
    void update_and_display_fps(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta);

    /**
     * @brief Renders the OSD for a non-selected object.
     * Sets the bounding box color to blue and displays standard object information.
     *
     * @param obj_meta Pointer to the NvDsObjectMeta for the current non-selected object.
     */
    void render_non_selected_object_osd(NvDsObjectMeta *obj_meta);

    /**
     * @brief Renders the OSD for the selected object, handling both detected and occluded states.
     * This function draws the appropriate bounding box (DeepStream's or KF-predicted),
     * reticule, and detailed text based on the object's detection status.
     *
     * @param batch_meta Pointer to the current NvDsBatchMeta.
     * @param frame_meta Pointer to the current NvDsFrameMeta.
     * @param selected_object_id The ID of the currently selected object.
     * @param selected_object_kf_initialized True if the Kalman Filter for the selected object is initialized.
     * @param selected_object_last_bbox The last known (or KF-predicted) bounding box of the selected object.
     * @param selected_object_lost_frames The number of frames the selected object has been lost by the detector.
     * @param selected_object_tracker_state The DeepStream tracker state of the selected object.
     * @param predicted_vx The predicted X velocity from the Kalman Filter for the selected object.
     * @param predicted_vy The predicted Y velocity from the Kalman Filter for the selected object.
     * @param is_selected_object_currently_detected True if the selected object was found in the current DeepStream detections.
     * @param selected_obj_meta_ptr Pointer to the NvDsObjectMeta of the selected object if it was detected, nullptr otherwise.
     */
    void render_selected_object_osd(
        NvDsBatchMeta *batch_meta,
        NvDsFrameMeta *frame_meta,
        guint64 selected_object_id,
        bool selected_object_kf_initialized,
        const NvOSD_RectParams &selected_object_last_bbox,
        unsigned int selected_object_lost_frames,
        TRACKER_STATE selected_object_tracker_state,
        double predicted_vx, double predicted_vy,
        bool is_selected_object_currently_detected,
        NvDsObjectMeta *selected_obj_meta_ptr);

private:
    /**
     * @brief Draws a reticule (crosshair) at a specified center point on the OSD.
     * @param batch_meta The NvDsBatchMeta for acquiring display metadata.
     * @param frame_meta The NvDsFrameMeta to add display metadata to.
     * @param center_x The X coordinate of the reticule's center.
     * @param center_y The Y coordinate of the reticule's center.
     * @param size The size (length of each arm) of the reticule.
     * @param color The color of the reticule.
     */
    void draw_reticule(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                       gfloat center_x, gfloat center_y, gfloat size, NvOSD_ColorParams color);

    /**
     * @brief Helper to set common text parameters for OSD labels.
     * @param text_params Pointer to the NvOSD_TextParams structure to modify.
     * @param x_offset X coordinate for the text.
     * @param y_offset Y coordinate for the text.
     * @param font_color Color for the text.
     * @param display_text The string to display.
     */
    void set_text_params(
        NvOSD_TextParams *text_params,
        guint x_offset, guint y_offset,
        NvOSD_ColorParams font_color,
        const std::string &display_text);


    // --- Members for FPS calculation and display ---
    std::chrono::steady_clock::time_point last_fps_update_time_; ///< Stores the last time FPS was updated.
    unsigned int frame_counter_;                                  ///< Counts frames since last FPS update.
    double current_fps_display_;                                  ///< Stores the calculated FPS value to be displayed.
    std::mutex fps_mutex_;                                        ///< Mutex to protect FPS variables from concurrent access.
};

#endif // OSD_RENDERER_HPP
