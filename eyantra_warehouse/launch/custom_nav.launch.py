#!/usr/bin/env python3
"""
Launch file for custom navigation with map server.
This file orchestrates the startup of:
1. Map Server - publishes the static map and map->odom transform
2. EKF - fuses odometry and IMU for accurate localization
3. Custom Navigator - our custom waypoint navigation system
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    eyantra_warehouse_dir = get_package_share_directory('eyantra_warehouse')
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    map_file_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(eyantra_warehouse_dir, 'maps', 'map.yaml'),
        description='Full path to map file to load'
    )
    
    # Use launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_file = LaunchConfiguration('map')
    
    # ============================================================
    # Step 1: Start Map Server (immediately)
    # ============================================================
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'yaml_filename': map_file}
        ]
    )
    
    # ============================================================
    # Step 2: Start EKF (after 1 second delay)
    # Requires odom from Gazebo/robot and imu data
    # ============================================================
    ekf_config_path = os.path.join(eyantra_warehouse_dir, 'config', 'ekf.yaml')
    
    ekf_node = TimerAction(
        period=1.0,
        actions=[
            Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_filter_node',
                output='screen',
                parameters=[ekf_config_path, {'use_sim_time': use_sim_time}]
            )
        ]
    )
    
    # ============================================================
    # Step 3: Start Custom Navigator (after 3 seconds delay)
    # This allows time for:
    # - Map server to publish map frame
    # - EKF to fuse initial odometry/IMU
    # - TF tree to stabilize
    # ============================================================
    custom_nav_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='eyantra_warehouse',
                executable='custom_nav.py',
                name='custom_navigator',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        map_file_arg,
        map_server_node,      # 0s
        ekf_node,             # 1s
        custom_nav_node,      # 3s
    ])
