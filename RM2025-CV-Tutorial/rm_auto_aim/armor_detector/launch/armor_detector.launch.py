import os
import sys
import yaml

from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch.actions import TimerAction, Shutdown
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():

    node_params = os.path.join(
        get_package_share_directory("armor_detector"),
        "config",
        "armor_detector.yaml",
    )

    armor_detector_node = Node(
        package="armor_detector",
        executable="armor_detector_node",
        emulate_tty=True,
        output="both",
        parameters=[node_params],
        arguments=["--ros-args", "--log-level", "armor_detector_node:=" + "INFO"],
    )

    return LaunchDescription(
        [
            armor_detector_node,
        ]
    )