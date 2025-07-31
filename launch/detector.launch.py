import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the path to your package's share directory
    pkg_share_dir = get_package_share_directory('ros2_object_detection')

    # Path to the main parameters YAML file
    parameters_file_path = os.path.join(pkg_share_dir, 'config', 'object_detection_node.yaml')

    # The current working directory for the node.
    # This is important for DeepStream config files (like pgie_yolov11_config.txt
    # and tracker_config.yaml) if their paths in the pipeline_string are relative.
    # Assuming 'inference' and 'tracking' directories are directly under your
    # package's share directory.
    node_cwd = pkg_share_dir

    return LaunchDescription([
        Node(
            package='ros2_object_detection',
            executable='object_detection_node',
            name='object_detection_node',
            output='screen',
            # Load all parameters from the full_params.yaml file
            parameters=[parameters_file_path],
            # Set the current working directory for the node process.
            # This ensures relative paths in the GStreamer pipeline string
            # (e.g., 'inference/pgie_yolov11_config.txt') are resolved correctly.
            cwd=node_cwd,
            arguments=['--ros-args', '--log-level', 'info']
        )
    ])
