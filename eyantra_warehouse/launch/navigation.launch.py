# mobilerobo/launch/nav.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package directory
    mobilerobo_pkg_dir = get_package_share_directory("eyantra_warehouse")
    # nav2_bringup_pkg_dir = get_package_share_directory('nav2_bringup')
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(mobilerobo_pkg_dir, 'config', 'nav2-config.yaml'),
        description='Full path to the ROS2 parameters file for Nav2'
    )

    map_file_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(mobilerobo_pkg_dir, 'maps', 'house_map.yaml'),
        description='Full path to map file to load'
    )

    # Use launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    map_file = LaunchConfiguration('map')

    # Define the list of Nav2 lifecycle nodes to be managed
    lifecycle_nodes = [
        'map_server',
        'amcl',
        'controller_server',
        'planner_server',
        'bt_navigator'
    ]

    # --- Start Each Nav2 Node Individually ---

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'yaml_filename': map_file}]
    )

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[params_file]
    )

    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file]
    )

    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file]
    )

    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file]
    )

    # --- Lifecycle Manager to bring them all up ---

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': lifecycle_nodes}]
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        map_file_arg,
        
        # Add all the nodes to the launch description
        map_server_node,
        amcl_node,
        controller_server_node,
        planner_server_node,
        bt_navigator_node,
        lifecycle_manager_node
    ])
