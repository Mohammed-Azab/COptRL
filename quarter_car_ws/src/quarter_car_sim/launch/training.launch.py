"""
# ============================================================
# REAL TRAINING: python training/train.py --algo sac
#
# This launch file is ONLY for testing the ROS topic pipeline.
# It starts a passive sim_node (no controller, no Gazebo).
# ============================================================
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='quarter_car_sim',
            executable='sim_node',
            name='sim_node',
            parameters=[{
                'passive':      True,
                'road_profile': 'speed_bump',
            }],
            output='screen',
        ),
    ])
