#!/usr/bin/env python3
"""
Combined navigation launch file for ebot waypoint navigation
Runs after: 
  1. ros2 launch eyantra_warehouse task1.launch.py
  2. ros2 launch ebot_description spawn_ebot.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    pkg_share = FindPackageShare(package='eyantra_warehouse').find('eyantra_warehouse')
    
    # Config file paths
    ekf_config_path = os.path.join(pkg_share, 'config', 'ekf.yaml')
    nav2_config_path = os.path.join(pkg_share, 'config', 'nav2-config.yaml')
    map_file_path = os.path.join(pkg_share, 'maps', 'map.yaml')
    rviz_config_file = os.path.join(pkg_share, 'config', 'nav.rviz')
    
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # Nav2 lifecycle nodes list
    lifecycle_nodes = [
        'map_server',
        'amcl',
        'controller_server',
        'planner_server',
        'bt_navigator'
    ]
    
    # Step 1: Start EKF node (immediately)
    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    # Step 2: Start Map Server (2s delay)
    map_server_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='nav2_map_server',
                executable='map_server',
                name='map_server',
                output='screen',
                parameters=[
                    {'use_sim_time': LaunchConfiguration('use_sim_time')},
                    {'yaml_filename': map_file_path}
                ]
            )
        ]
    )
    
    # Step 3: Start AMCL (3s delay)
    amcl_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='nav2_amcl',
                executable='amcl',
                name='amcl',
                output='screen',
                parameters=[nav2_config_path]
            )
        ]
    )
    
    # Step 4: Start Controller Server (4s delay)
    controller_server_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                output='screen',
                parameters=[nav2_config_path]
            )
        ]
    )
    
    # Step 5: Start Planner Server (5s delay)
    planner_server_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                parameters=[nav2_config_path]
            )
        ]
    )
    
    # Step 6: Start BT Navigator (6s delay)
    bt_navigator_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                parameters=[nav2_config_path]
            )
        ]
    )
    
    # Step 7: Start Lifecycle Manager (7s delay)
    lifecycle_manager_node = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                output='screen',
                parameters=[
                    {'use_sim_time': LaunchConfiguration('use_sim_time')},
                    {'autostart': True},
                    {'node_names': lifecycle_nodes}
                ]
            )
        ]
    )
    
    # Step 8: Start RViz (8s delay)
    rviz_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config_file],
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
            )
        ]
    )
    
    # Step 9: Start Navigation Goals sender (12s delay - after all Nav2 is ready)
    navigation_goals_node = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='eyantra_warehouse',
                executable='navigation_goals.py',
                name='navigation_goal_sender',
                output='screen',
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
            )
        ]
    )
    
    return LaunchDescription([
        use_sim_time,
        robot_localization_node,      # 0s - EKF
        map_server_node,              # 2s - Map Server
        amcl_node,                    # 3s - AMCL
        controller_server_node,       # 4s - Controller
        planner_server_node,          # 5s - Planner
        bt_navigator_node,            # 6s - BT Navigator
        lifecycle_manager_node,       # 7s - Lifecycle Manager
        rviz_node,                    # 8s - RViz
        navigation_goals_node,        # 12s - Start sending waypoints
    ])
