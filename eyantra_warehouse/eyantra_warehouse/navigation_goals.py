#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
import math

class NavigationGoalSender(Node):
    def __init__(self):
        super().__init__('navigation_goal_sender')
        
        # Create action client for NavigateToPose
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Define poses: [x, y, yaw]
        self.poses = [
            [-1.53, -1.95, 1.57],
            [0.13, 1.24, 0.00],
            [0.38, -3.32, -1.57]
        ]
        
        self.current_pose_index = 0
        
        self.get_logger().info('Navigation Goal Sender initialized')
        self.get_logger().info('Waiting for action server...')
        
        # Wait for action server
        self._action_client.wait_for_server()
        self.get_logger().info('Action server available!')
        
        # Send first goal
        self.send_next_goal()
    
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
        
        self.get_logger().info(
            f'Sending goal {self.current_pose_index + 1}/{len(self.poses)}: '
            f'P{self.current_pose_index + 1} = [{x:.2f}, {y:.2f}, {yaw:.2f}]'
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
            return
        
        self.get_logger().info('Goal accepted by action server')
        
        # Wait for result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def feedback_callback(self, feedback_msg):
        """Callback for feedback from action server"""
        feedback = feedback_msg.feedback
        # Get distance remaining
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
            # Move to next pose
            self.current_pose_index += 1
            self.send_next_goal()
        else:
            self.get_logger().error(
                f'Goal {self.current_pose_index + 1} failed with status: {status}'
            )
            self.get_logger().info('Stopping navigation sequence')


def main(args=None):
    rclpy.init(args=args)
    
    navigation_goal_sender = NavigationGoalSender()
    
    try:
        rclpy.spin(navigation_goal_sender)
    except KeyboardInterrupt:
        pass
    
    navigation_goal_sender.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
