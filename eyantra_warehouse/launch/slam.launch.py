import os
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Get the package directory
    pkg_share = FindPackageShare(package='eyantra_warehouse').find('eyantra_warehouse')
    robot_discription_directory = get_package_share_directory('ebot_description')
    # controller_config = os.path.join(pkg_share, 'config', 'controllers.yaml')
    slam_config = os.path.join(pkg_share, 'config', 'slam-config.yaml')
    ekf_config_path = os.path.join(pkg_share, 'config', 'ekf.yaml')
    
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # rviz node
    rviz_config_file = os.path.join(pkg_share, 'config', 'robot.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # # ROS-GZ Bridge
    # gz_bridge_config = os.path.join(pkg_share, 'config', 'gz-bridge.yaml')
    # bridge_cmd = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     parameters=[{
    #         'config_file': gz_bridge_config
    #     }],
    #     output='screen'
    # )
 
    # Include robot-spawn.launch.py to spawn the robot in Gazebo
    # robot_spawn_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(pkg_share, 'launch', 'robot-spawn.launch.py')
    #     ),
    #     launch_arguments={
    #         'use_sim_time': LaunchConfiguration('use_sim_time'),
    #         'x': LaunchConfiguration('x'),
    #         'y': LaunchConfiguration('y')
    #     }.items()
    # )

    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    slam_toolbox_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'slam_params_file': slam_config
        }.items()
    )

    return LaunchDescription([
        use_sim_time,
        robot_localization_node,
        slam_toolbox_node,
        rviz_node,
    ])
