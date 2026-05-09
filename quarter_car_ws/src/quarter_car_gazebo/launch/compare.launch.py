"""
Compare launch: RL + LQR stub + MPC stub running simultaneously.
Note: lqr_node and mpc_node are stubs — only RL shows real control.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path = LaunchConfiguration('model_path')
    algo       = LaunchConfiguration('algo')

    return LaunchDescription([
        DeclareLaunchArgument('model_path', description='Path to SB3 .zip model'),
        DeclareLaunchArgument('algo', default_value='sac'),

        Node(package='quarter_car_gazebo',
             executable='sim_node',
             parameters=[{'passive': False}],
             output='screen'),

        Node(package='quarter_car_controllers',
             executable='rl_node',
             parameters=[{'model_path': model_path, 'algo': algo}],
             output='screen'),

        Node(package='quarter_car_controllers',
             executable='lqr_node',
             output='screen'),

        Node(package='quarter_car_controllers',
             executable='mpc_node',
             output='screen'),
    ])
