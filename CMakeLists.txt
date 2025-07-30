cmake_minimum_required(VERSION 3.8)
project(ros2_object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find necessary packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED) # This finds Eigen and sets EIGEN3_INCLUDE_DIRS
find_package(yaml-cpp REQUIRED)

# Find GStreamer, GLib, and yaml-cpp
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)

# Find DeepStream libraries
find_library(NVDS_META_LIB
  NAMES nvds_meta
  PATHS /opt/nvidia/deepstream/deepstream-7.1/lib/
  NO_DEFAULT_PATH
)
if(NOT NVDS_META_LIB)
  message(FATAL_ERROR "DeepStream nvds_meta library not found! Please ensure libnvds_meta.so is in /opt/nvidia/deepstream/deepstream-7.1/lib/")
endif()

find_library(GST_NVDS_META_LIB
  NAMES nvdsgst_meta
  PATHS /opt/nvidia/deepstream/deepstream-7.1/lib/
  NO_DEFAULT_PATH
)
if(NOT GST_NVDS_META_LIB)
  message(FATAL_ERROR "DeepStream gstnvdsmeta library not found! Please ensure libgstnvdsmeta.so is in /opt/nvidia/deepstream/deepstream-7.1/lib/")
endif()

# Build the C++ node executable
add_executable(object_detection_node
  src/object_detection.cpp
  src/kalman_filter_2d.cpp # <--- IMPORTANT: Added this line!
)

# Add dependencies for your node
ament_target_dependencies(object_detection_node
  rclcpp
  yaml-cpp
  vision_msgs
  cv_bridge
  sensor_msgs
  image_transport
  Eigen3 # Keep this here, it helps with ROS dependency tracking
)

# Add the include directories so the compiler can find all necessary headers
target_include_directories(object_detection_node PUBLIC
  include
  ${GSTREAMER_INCLUDE_DIRS}
  ${GLIB_INCLUDE_DIRS}
  /opt/nvidia/deepstream/deepstream/sources/includes/
  ${YAML_CPP_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Link against all required libraries
target_link_libraries(object_detection_node
  ${GSTREAMER_LIBRARIES}
  ${GLIB_LIBRARIES}
  ${NVDS_META_LIB}
  ${GST_NVDS_META_LIB}
  ${YAML_CPP_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  # Eigen is typically header-only, so explicit linking here is often not needed,
  # but if you use Eigen's dense linear algebra features, it might be.
  # If you encounter linking errors related to Eigen, uncomment the line below.
  # ${Eigen3_LIBS} # Uncomment if you get Eigen linking errors
)

# Install targets
install(TARGETS
  object_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()