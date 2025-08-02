import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable

def generate_launch_description():
    # Get the path to your package's share directory
    pkg_share_dir = get_package_share_directory('ros2_object_detection')

    # Path to the common parameters file
    parameters_file_path = os.path.join(pkg_share_dir, 'config', 'object_detection_node.yaml')

    # The current working directory for the node.
    # This is important for DeepStream config files if their paths are relative.
    node_cwd = pkg_share_dir

    # Define GStreamer debug level
    gst_debug_level = "nvtracker:6"

    # --- Node for Object Detection (Perception) ---
    object_detection_node = Node(
        package='ros2_object_detection',
        executable='object_detection_node',
        name='object_detection_node',
        output='screen',
        parameters=[parameters_file_path],
        cwd=node_cwd,
        arguments=['--ros-args', '--log-level', 'info']
    )

    # --- Node for Target Logic (The "Brain") ---
    target_logic_node = Node(
        package='ros2_object_detection',
        executable='target_logic_node',
        name='object_targetting_node',
        output='screen',
        parameters=[parameters_file_path],
        arguments=['--ros-args', '--log-level', 'info']
    )

    return LaunchDescription([
        SetEnvironmentVariable(name='GST_DEBUG', value=gst_debug_level),
        object_detection_node,
        target_logic_node
    ])
