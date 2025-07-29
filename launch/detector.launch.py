import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share_dir = get_package_share_directory('ros2_object_detection')
    config_file = os.path.join(pkg_share_dir, 'config', 'pipelines', 'test_pipeline.yaml')

    # This is the directory where your config files are installed
    config_dir = os.path.join(pkg_share_dir, 'config')

    return LaunchDescription([
        Node(
            package='ros2_object_detection',
            executable='object_detection_node',
            name='object_detection_node',
            output='screen',
            parameters=[{'pipeline_config': config_file}],
            cwd=config_dir
        )
    ])