# Tests the ROS topic pipeline only — passive sim_node, no controller, no Gazebo.
# For actual training: python training/train.py --algo sac
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='quarter_car_gazebo',
            executable='sim_node',
            name='sim_node',
            parameters=[{
                'passive':      True,
                'road_profile': 'speed_bump',
            }],
            output='screen',
        ),
    ])
