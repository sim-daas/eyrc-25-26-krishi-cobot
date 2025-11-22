#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' 
*****************************************************************************************
*  Filename:       custom_nav.py
*  Description:    Custom navigation node for the ebot
*  Modified by:    Sahil
*  Author:         e-Yantra Team
*****************************************************************************************
'''

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import math
import time

class CustomNavigator(Node):
    """Custom navigation node for structured environment navigation."""
    
    CONTROL_LOOP_RATE = 10  # Control loop rate in Hz
    
    def __init__(self):
        super().__init__('custom_navigator')
        
        # ============================================================
        # ROS Setup
        # ============================================================
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/ebot/global_path', 10)
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/ebot/pose',
            self.pose_callback,
            10
        )
        
        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.CONTROL_LOOP_RATE, self.navigation_loop
        )
        
        # ============================================================
        # Initialization
        # ============================================================
        
        self.get_logger().info('Custom Navigator initialized')
        self.get_logger().info(f'Mission sequence: {self.MISSION_SEQUENCE}')
        
        # Defer TF lookup until the TF tree actually exists
        self.state = 'WAITING_FOR_TF'
        self.wait_for_tf_start = self.get_clock().now()
        self.tf_ready = False
        self.get_logger().info('Waiting for TF tree (map -> odom -> ebot_base_link)...')
        
        # Initialize path and mission indices
        self.global_path = []
        self.path_index = 0
        self.mission_index = 0
        self.wait_start_time = None

    def pose_callback(self, msg):
        # Update the current position of the robot
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
    
    def scan_callback(self, msg):
        # TODO: Implement obstacle avoidance and navigation logic
        pass
    
    def navigation_loop(self):
        if not self.tf_ready:
            if self.try_initialize_tf():
                if self.plan_mission():
                    self.tf_ready = True
                    self.state = 'TURNING_TO_NEXT'
                    self.get_logger().info(f'✓ Mission started! Path: {" -> ".join(self.global_path)}')
                else:
                    return
            else:
                return
        
        # TODO: Handle different states (e.g., TURNING_TO_NEXT, MOVING_TO_NEXT, etc.)
    
    def try_initialize_tf(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'ebot_base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )
            self.current_x = transform.transform.translation.x
            self.current_y = transform.transform.translation.y
            self.current_yaw = self.quaternion_to_yaw(transform.transform.rotation)
            self.get_logger().info(
                f'✓ TF tree ready! Robot pose: x={self.current_x:.3f}, y={self.current_y:.3f}, '
                f'yaw={math.degrees(self.current_yaw):.1f}°'
            )
            return True
        except Exception:
            elapsed = (self.get_clock().now() - self.wait_for_tf_start).nanoseconds / 1e9
            if int(elapsed) % 5 == 0:
                self.get_logger().warn('Still waiting for TF tree...', throttle_duration_sec=5.0)
            return False

    def plan_mission(self):
        if not self.update_robot_pose():
            self.get_logger().warn('Robot pose unavailable yet; will retry.', throttle_duration_sec=2.0)
            return False
        # TODO: Implement mission planning logic
        return True
    
    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle."""
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return yaw

def main(args=None):
    rclpy.init(args=args)
    
    try:
        navigator = CustomNavigator()
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if 'navigator' in locals():
            navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()