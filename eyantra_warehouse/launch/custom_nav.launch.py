#!/usr/bin/env python3
"""
Launch file for custom navigation with map server.
This file orchestrates the startup of:
1. Map Server - publishes the static map
2. Lifecycle Manager - activates the map server
3. Static TF Publisher - publishes map->odom transform
4. EKF - fuses odometry and IMU, publishes odom->ebot_base_link
5. Custom Navigator - our custom waypoint navigation system

TF Tree: map -> odom -> ebot_base_link
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
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
    # Lifecycle node - needs to be configured and activated
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
    # Step 2: Lifecycle Manager for Map Server (1s delay)
    # This will configure and activate the map server
    # ============================================================
    lifecycle_manager_node = TimerAction(
        period=1.0,
        actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_localization',
                output='screen',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server']}
                ]
            )
        ]
    )
    
    # ============================================================
    # Step 3: Static TF Publisher for map->odom (2s delay)
    # Since robot spawns at map origin (0,0,0)
    # ============================================================
    static_tf_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='map_to_odom_publisher',
                output='screen',
                arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )
    
    # ============================================================
    # Step 4: Start EKF (3s delay)
    # Fuses wheel odometry and IMU
    # Publishes odom -> ebot_base_link transform
    # ============================================================
    ekf_config_path = os.path.join(eyantra_warehouse_dir, 'config', 'ekf.yaml')
    
    ekf_node = TimerAction(
        period=3.0,
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
    # Step 5: Start Custom Navigator (7s delay - increased from 5s)
    # Wait for:
    # - Map server to be activated and publish /map (done by ~2s)
    # - Static TF to publish map -> odom (done by ~2s)
    # - EKF to start and publish odom -> ebot_base_link (takes ~4s)
    # - Complete TF tree: map -> odom -> ebot_base_link
    # The navigator itself will wait up to 30s for TF if needed
    # ============================================================
    custom_nav_node = TimerAction(
        period=7.0,  # Increased from 5s to 7s
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
        map_server_node,          # 0s - Map Server (lifecycle)
        lifecycle_manager_node,   # 1s - Activate Map Server
        static_tf_node,           # 2s - Static TF: map -> odom
        ekf_node,                 # 3s - EKF: odom -> ebot_base_link
        custom_nav_node,          # 7s - Custom Navigator (increased delay)
    ])
