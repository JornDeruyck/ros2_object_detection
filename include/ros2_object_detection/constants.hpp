// include/ros2_object_detection/constants.hpp
#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// #include <glib.h> // For G_MAXUINT64 - NOT NEEDED IF USING DEEPSTREAM'S UNTRACKED_OBJECT_ID

// Define the threshold for how many frames an object can be lost by our KF before deselection
// This counter is distinct from the DeepStream tracker's internal "lost" state.
const unsigned int KF_LOST_THRESHOLD = 100; // How many frames our custom Kalman Filter will predict without new measurements before declaring the object lost and deselecting.

// Define DeepStream tracker metadata type (if not reliably defined elsewhere)
// Typically, this is defined in nvds_tracker_meta.h, but adding it here for robustness if needed.
#ifndef NVDS_TRACKER_METADATA
#define NVDS_TRACKER_METADATA 103
#endif

#endif // CONSTANTS_HPP