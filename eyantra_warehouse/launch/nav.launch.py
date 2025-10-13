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
    pkg_share = FindPackageShare(package='mobilerobo').find('mobilerobo')
    robot_discription_directory = get_package_share_directory('robobase_description')
    controller_config = os.path.join(pkg_share, 'config', 'controllers.yaml')
    slam_config = os.path.join(pkg_share, 'config', 'slam-config.yaml')
    ekf_config_path = os.path.join(pkg_share, 'config', 'ekf.yaml')
    twist_mux_config = os.path.join(pkg_share, 'config', 'twist_mux.yaml')
    # Path to the URDF xacro file
    urdf_file = os.path.join(robot_discription_directory, 'models', 'robo.urdf.xacro')
    urdf_xacro = os.path.join(robot_discription_directory, 'models', 'robo.urdf.xacro')

    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # rviz node
    rviz_config_file = os.path.join(pkg_share, 'config', 'nav.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # start the nav2 nodes
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'navigation.launch.py')
        ),
    )

    return LaunchDescription([
        use_sim_time,
        robot_localization_node,
        navigation_launch,
        rviz_node,
    ])
