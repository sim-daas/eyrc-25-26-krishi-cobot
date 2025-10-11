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
    robot_discription_directory = get_package_share_directory('ebot_description')
    controller_config = os.path.join(pkg_share, 'config', 'controllers.yaml')
    slam_config = os.path.join(pkg_share, 'config', 'slam-config.yaml')
    ekf_config_path = os.path.join(pkg_share, 'config', 'ekf.yaml')
    twist_mux_config = os.path.join(pkg_share, 'config', 'twist_mux.yaml')
    # Path to the URDF xacro file
    urdf_file = os.path.join(robot_discription_directory, 'models', 'robo.urdf.xacro')
    urdf_xacro = os.path.join(robot_discription_directory, 'models', 'robo.urdf.xacro')

    # World file path
    world_file = os.path.join(pkg_share, 'worlds', 'house.world')
    
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    x = DeclareLaunchArgument(
        'x',
        default_value='0.0',
        description='X position of the robot in the world'
    )
    
    y = DeclareLaunchArgument(
        'y',
        default_value='-5.0',
        description='Y position of the robot in the world'
    )
   
    # Set GZ_RESOURCE_PATH to include models directory
    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(robot_discription_directory, 'models')
    )

    # Launch GZ Sim server with world
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world_file], 'on_exit_shutdown': 'true'}.items()
    )

    # Launch GZ Sim GUI
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 ', 'on_exit_shutdown': 'true'}.items()
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

    # ROS-GZ Bridge
    gz_bridge_config = os.path.join(pkg_share, 'config', 'gz-bridge.yaml')
    bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': gz_bridge_config
        }],
        output='screen'
    )
 
    # Include robot-spawn.launch.py to spawn the robot in Gazebo
    robot_spawn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'robot-spawn.launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y')
        }.items()
    )

    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )


    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )

    # Load and start the Differential Drive Controller
    diff_drive_base_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller", "--controller-manager", "/controller_manager"],
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

    odom_relay_node = Node(
        package='topic_tools',
        executable='relay',
        name='odom_relay',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=['/odometry/filtered', '/odom']
    )

    # TwistMux node
    twist_mux = Node(
        package="twist_mux",
        executable="twist_mux",
        name='twist_mux',
        parameters=[twist_mux_config, {"use_sim_time": True}],
        remappings=[('/cmd_vel_out', '/diff_drive_controller/cmd_vel')],
    )

    # Include joystick.launch.py
    joystick_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'joystick.launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )

    
    return LaunchDescription([
        use_sim_time,
        x,
        y,
        set_env_vars_resources,
        gzserver_cmd,
        gzclient_cmd,
        robot_spawn_launch,
        robot_localization_node,
        joint_state_broadcaster_spawner,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[diff_drive_base_controller_spawner],
            )
        ),
        odom_relay_node,
        slam_toolbox_node,
        rviz_node,
        bridge_cmd,
        twist_mux,
        joystick_launch,
    ])
