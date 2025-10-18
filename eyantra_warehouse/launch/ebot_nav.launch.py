#!/usr/bin/env python3
"""
Combined navigation launch file for ebot waypoint navigation
Includes all Nav2 nodes and embedded waypoint navigation logic
Runs after: 
  1. ros2 launch eyantra_warehouse task1.launch.py
  2. ros2 launch ebot_description spawn_ebot.launch.py
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
from lifecycle_msgs.srv import GetState
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node as LaunchNode
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


class NavigationGoalSender(Node):
    """Embedded navigation goal sender within launch file"""
    def __init__(self):
        super().__init__('navigation_goal_sender')
        
        # Create action client for NavigateToPose
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Robot's origin position in world coordinates
        self.world_origin_x = -1.5339
        self.world_origin_y = -6.6156
        self.world_origin_yaw = 1.57
        
        # Define poses in world coordinates: [x, y, yaw]
        world_poses = [
            [-1.53, -1.95, 1.57],
            [0.13, 1.24, 0.00],
            [0.38, -3.32, -1.57]
        ]
        
        # Convert world coordinates to map coordinates
        self.poses = []
        for wx, wy, wyaw in world_poses:
            # Swap X and Y due to coordinate frame rotation, negate Y
            map_x = wy - self.world_origin_y
            map_y = -(wx - self.world_origin_x)
            map_yaw = wyaw - self.world_origin_yaw
            self.poses.append([map_x, map_y, map_yaw])
            self.get_logger().info(
                f'World pose [{wx:.2f}, {wy:.2f}, {wyaw:.2f}] -> '
                f'Map pose [{map_x:.2f}, {map_y:.2f}, {map_yaw:.2f}]'
            )
        
        self.current_pose_index = 0
        self.retry_count = 0
        self.max_retries = 3
        
        self.get_logger().info('Navigation Goal Sender initialized')
        self.get_logger().info('Waiting for Nav2 to be fully active...')
        
        # Wait for Nav2 to be ready
        self.wait_for_nav2_ready()
    
    def wait_for_nav2_ready(self):
        """Wait for bt_navigator to be in active state before sending goals"""
        # Create client to check bt_navigator state
        state_client = self.create_client(GetState, '/bt_navigator/get_state')
        
        # Wait for service to be available
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for bt_navigator lifecycle service...')
        
        # Check state periodically
        timer = self.create_timer(1.0, lambda: self.check_nav2_state(state_client))
    
    def check_nav2_state(self, state_client):
        """Check if bt_navigator is active"""
        request = GetState.Request()
        future = state_client.call_async(request)
        future.add_done_callback(self.state_callback)
    
    def state_callback(self, future):
        """Callback when state is received"""
        try:
            response = future.result()
            # State 3 = ACTIVE in lifecycle
            if response.current_state.id == 3:
                self.get_logger().info('Nav2 is fully active! Waiting for action server...')
                # Cancel the timer and proceed
                for timer in self.get_timers():
                    timer.cancel()
                
                # Now wait for action server
                self._action_client.wait_for_server()
                self.get_logger().info('Action server available!')
                
                # Send first goal
                self.send_next_goal()
            else:
                self.get_logger().info(
                    f'Nav2 state: {response.current_state.label} (waiting for active...)'
                )
        except Exception as e:
            self.get_logger().warn(f'Failed to get state: {e}')
    
    def create_pose_stamped(self, x, y, yaw):
        """Create a PoseStamped message from x, y, yaw"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        quaternion = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        
        return pose
    
    def send_next_goal(self):
        """Send the next navigation goal"""
        if self.current_pose_index >= len(self.poses):
            self.get_logger().info('All goals completed successfully!')
            return
        
        # Get current pose
        x, y, yaw = self.poses[self.current_pose_index]
        
        retry_msg = f' (Retry {self.retry_count}/{self.max_retries})' if self.retry_count > 0 else ''
        self.get_logger().info(
            f'Sending goal {self.current_pose_index + 1}/{len(self.poses)}: '
            f'P{self.current_pose_index + 1} = [{x:.2f}, {y:.2f}, {yaw:.2f}]{retry_msg}'
        )
        
        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.create_pose_stamped(x, y, yaw)
        
        # Send goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Callback for when goal is accepted or rejected"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server!')
            self.handle_goal_failure()
            return
        
        self.get_logger().info('Goal accepted by action server')
        
        # Wait for result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def feedback_callback(self, feedback_msg):
        """Callback for feedback from action server"""
        feedback = feedback_msg.feedback
        distance_remaining = feedback.distance_remaining
        self.get_logger().info(
            f'Distance remaining: {distance_remaining:.2f}m',
            throttle_duration_sec=2.0
        )
    
    def get_result_callback(self, future):
        """Callback for when goal is completed"""
        result = future.result().result
        status = future.result().status
        
        if status == 4:  # SUCCEEDED
            self.get_logger().info(
                f'Goal {self.current_pose_index + 1} reached successfully!'
            )
            # Reset retry count and move to next pose
            self.retry_count = 0
            self.current_pose_index += 1
            self.send_next_goal()
        else:
            self.get_logger().error(
                f'Goal {self.current_pose_index + 1} failed with status: {status}'
            )
            self.handle_goal_failure()
    
    def handle_goal_failure(self):
        """Handle goal failure with retry logic"""
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            self.get_logger().warn(
                f'Retrying goal {self.current_pose_index + 1} '
                f'(attempt {self.retry_count + 1}/{self.max_retries + 1})'
            )
            # Wait a bit before retrying
            self.create_timer(2.0, self.retry_current_goal)
        else:
            self.get_logger().error(
                f'Goal {self.current_pose_index + 1} failed after {self.max_retries} retries. '
                f'Stopping navigation sequence.'
            )
    
    def retry_current_goal(self):
        """Retry the current goal"""
        self.send_next_goal()


def start_navigation_goals(context):
    """Function to start the navigation goals node after Nav2 is ready"""
    rclpy.init()
    navigation_goal_sender = NavigationGoalSender()
    
    # Spin in a separate thread
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(navigation_goal_sender,), daemon=True)
    spin_thread.start()
    
    return []


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
    robot_localization_node = LaunchNode(
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
            LaunchNode(
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
            LaunchNode(
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
            LaunchNode(
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
            LaunchNode(
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
            LaunchNode(
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
            LaunchNode(
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
            LaunchNode(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config_file],
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
            )
        ]
    )
    
    # Step 9: Start embedded Navigation Goals (after RViz, will wait for Nav2 internally)
    start_nav_goals = TimerAction(
        period=10.0,  # Just wait for RViz to start, then check Nav2 state internally
        actions=[OpaqueFunction(function=start_navigation_goals)]
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
        start_nav_goals,              # 10s - Start goal sender (waits for Nav2 active state internally)
    ])
