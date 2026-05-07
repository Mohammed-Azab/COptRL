"""
Gazebo evaluation launch.
NO RViz2. Gazebo is the only 3D view. rqt_plot is the only signal view.

Usage:
  ros2 launch quarter_car_sim eval_gazebo.launch.py \
    model_path:=/path/to/model.zip algo:=sac
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('quarter_car_sim')

    model_path   = LaunchConfiguration('model_path')
    algo         = LaunchConfiguration('algo')
    road_profile = LaunchConfiguration('road_profile')

    return LaunchDescription([
        DeclareLaunchArgument('model_path',   description='Path to SB3 .zip model'),
        DeclareLaunchArgument('algo',         default_value='sac'),
        DeclareLaunchArgument('road_profile', default_value='speed_bump'),

        # Gazebo simulation — speed_bump_world.sdf
        Node(
            package='ros_gz_sim', executable='create',
            arguments=[
                '-world', 'speed_bump_world',
                '-file',  os.path.join(pkg, 'worlds', 'speed_bump_world.sdf'),
            ],
            output='screen',
        ),

        # Robot state publisher (processes URDF xacro)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description':
                    open(os.path.join(pkg, 'urdf',
                                     'quarter_car_robot.urdf.xacro')).read()
            }],
            output='screen',
        ),

        # ros_gz_bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['--ros-args', '--params-file',
                       os.path.join(pkg, 'config', 'gz_bridge.yaml')],
            output='screen',
        ),

        # Physics simulation node
        Node(
            package='quarter_car_sim',
            executable='sim_node',
            parameters=[{'road_profile': road_profile, 'passive': False}],
            output='screen',
        ),

        # Gazebo bridge node (TF + joint cmd_pos)
        Node(
            package='quarter_car_sim',
            executable='gazebo_bridge_node',
            output='screen',
        ),

        # RL controller node
        Node(
            package='quarter_car_controllers',
            executable='rl_node',
            parameters=[{'model_path': model_path, 'algo': algo}],
            output='screen',
        ),

        # rqt_plot signal view — NO RViz2 anywhere
        Node(
            package='rqt_plot',
            executable='rqt_plot',
            arguments=[
                '/car/acceleration/data',
                '/car/comfort_score/data',
                '/car/reward/data',
            ],
            output='screen',
        ),
    ])
