#!/usr/bin/env python3
"""
Task 2A Navigation Script
Implements waypoint navigation and handles pause/resume for shape detection.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
import math
import time
from geometry_msgs.msg import TransformStamped

class Task2ANavigator(Node):
    
    # ============================================================
    # CONSTANTS
    # ============================================================
    
    # Task 2A Waypoints (Global Frame: x, y, yaw)
    # Note: Yaw is in radians. 1.57 rad ~= 90 deg, -1.57 rad ~= -90 deg
    GLOBAL_WAYPOINTS = {
        'P1': {'pos': (0.26, -1.95), 'yaw': 1.57},
        'P2': {'pos': (-1.48, -0.67), 'yaw': -1.57},
        'P3': {'pos': (-1.53, -6.61), 'yaw': -1.57},
    }
    
    # Robot Start Pose in Global Frame (from task description)
    START_POSE = {'x': -1.5339, 'y': -6.6156, 'yaw': 1.57}
    
    MISSION_SEQUENCE = ['P1', 'P2', 'P3']
    
    # Navigation Parameters
    LINEAR_VEL = 0.3          # m/s
    ANGULAR_VEL = 0.5         # rad/s
    
    # PID Gains
    KP_LINEAR = 0.5
    KP_ANGULAR = 1.5
    
    # Tolerances
    WAYPOINT_TOLERANCE = 0.2  # meters
    YAW_TOLERANCE = 0.1       # radians (~5.7 degrees)
    
    CONTROL_LOOP_RATE = 10.0  # Hz
    
    def __init__(self):
        super().__init__('ebot_nav_task2a')
        
        # Declare and get use_sim_time parameter
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        if use_sim_time:
            self.get_logger().info('Using simulation time')
            
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.nav_control_sub = self.create_subscription(
            String, '/nav_control', self.nav_control_callback, 10
        )
        
        # TF Listener with larger cache for simulation time
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer
        self.create_timer(1.0 / self.CONTROL_LOOP_RATE, self.control_loop)
        
        # State
        self.state = 'IDLE'  # IDLE, NAVIGATING, ALIGNING, PAUSED, COMPLETED
        self.previous_state = 'IDLE' # To restore after pause
        self.current_waypoint_idx = 0
        
        # Robot Pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        # Transformed Waypoints (in Odom frame)
        self.waypoints = {}
        
        # TF initialization flag
        self.tf_ready = False
        
        self.get_logger().info('Task 2A Navigator Initialized')
        self.get_logger().info('Waiting for TF frames (will check in background)...')

    def transform_waypoints(self):
        """
        Transform global waypoints to odom frame.
        Assuming odom frame starts at (0,0,0) relative to robot's spawn.
        So we transform Global -> Robot_Start -> Odom.
        Actually, if Robot spawns at START_POSE, and Odom starts at 0,0,0:
        P_odom = Rotate(-Start_Yaw) * (P_global - Start_Pos)
        """
        sx = self.START_POSE['x']
        sy = self.START_POSE['y']
        syaw = self.START_POSE['yaw']
        
        cos_yaw = math.cos(-syaw)
        sin_yaw = math.sin(-syaw)
        
        self.get_logger().info('Transforming waypoints to Odom frame...')
        
        for name, data in self.GLOBAL_WAYPOINTS.items():
            gx, gy = data['pos']
            gyaw = data['yaw']
            
            # Translate
            dx = gx - sx
            dy = gy - sy
            
            # Rotate
            ox = dx * cos_yaw - dy * sin_yaw
            oy = dx * sin_yaw + dy * cos_yaw
            
            # Rotate Yaw
            oyaw = self.normalize_angle(gyaw - syaw)
            
            self.waypoints[name] = {'pos': (ox, oy), 'yaw': oyaw}
            self.waypoints[name] = {'pos': (ox, oy), 'yaw': oyaw}
            self.get_logger().info(f'{name}: Global({gx:.2f}, {gy:.2f}) -> Odom({ox:.2f}, {oy:.2f})')

    def wait_for_transform(self):
        """Wait for odom -> ebot_base_link transform."""
        try:
            # Use Time(0) to get latest available transform
            if self.tf_buffer.can_transform('odom', 'ebot_base_link', rclpy.time.Time()):
                self.get_logger().info('Transform available!', throttle_duration_sec=5.0)
                return True
            
            self.get_logger().warn('Transform odom -> ebot_base_link not available yet', throttle_duration_sec=2.0)
            return False
        except Exception as e:
            self.get_logger().warn(f'Waiting for transform: {e}', throttle_duration_sec=2.0)
            return False

    def nav_control_callback(self, msg):
        """Handle external control commands (PAUSE/RESUME)."""
        command = msg.data
        if command == 'PAUSE':
            if self.state != 'PAUSED' and self.state != 'COMPLETED':
                self.get_logger().info('Pausing navigation for detection...')
                self.previous_state = self.state
                self.state = 'PAUSED'
                self.stop_robot()
        elif command == 'RESUME':
            if self.state == 'PAUSED':
                self.get_logger().info('Resuming navigation...')
                self.state = self.previous_state

    def update_pose(self):
        """Get current robot pose from TF."""
        try:
            # Use Time(0) to get latest available transform
            trans = self.tf_buffer.lookup_transform('odom', 'ebot_base_link', rclpy.time.Time())
            
            self.robot_x = trans.transform.translation.x
            self.robot_y = trans.transform.translation.y
            
            # Quaternion to Yaw
            q = trans.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
            return True
        except Exception as e:
            self.get_logger().warn(f'Could not get pose: {e}', throttle_duration_sec=1.0)
            return False

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def control_loop(self):
        # First-time TF initialization
        if not self.tf_ready:
            if self.wait_for_transform():
                self.get_logger().info('TF frames received! Initializing navigation...')
                self.transform_waypoints()
                self.state = 'NAVIGATING'
                self.tf_ready = True
            return
        
        if not self.update_pose():
            return

        if self.state == 'PAUSED':
            self.stop_robot()
            return
            
        if self.state == 'COMPLETED':
            self.stop_robot()
            return

        if self.current_waypoint_idx >= len(self.MISSION_SEQUENCE):
            self.get_logger().info('All waypoints completed!')
            self.state = 'COMPLETED'
            return

        # Get current target
        target_id = self.MISSION_SEQUENCE[self.current_waypoint_idx]
        target = self.waypoints[target_id]
        
        # Calculate errors
        dx = target['pos'][0] - self.robot_x
        dy = target['pos'][1] - self.robot_y
        dist = math.hypot(dx, dy)
        
        target_heading = math.atan2(dy, dx)
        heading_error = self.normalize_angle(target_heading - self.robot_yaw)

        # Navigation Logic
        cmd = Twist()

        if self.state == 'NAVIGATING':
            # If close enough to waypoint, switch to alignment
            if dist < self.WAYPOINT_TOLERANCE:
                self.get_logger().info(f'Reached {target_id} position. Aligning orientation...')
                self.state = 'ALIGNING'
                return

            # Simple Proportional Control
            # Turn to face target
            if abs(heading_error) > 0.1: # Turn in place if heading error is large
                cmd.angular.z = self.KP_ANGULAR * heading_error
                cmd.linear.x = 0.0
            else:
                # Drive forward and turn slightly
                cmd.linear.x = min(self.LINEAR_VEL, self.KP_LINEAR * dist)
                cmd.angular.z = self.KP_ANGULAR * heading_error
            
            # Clamp velocities
            cmd.linear.x = max(0.0, min(cmd.linear.x, self.LINEAR_VEL))
            cmd.angular.z = max(-self.ANGULAR_VEL, min(cmd.angular.z, self.ANGULAR_VEL))
            
            self.cmd_vel_pub.publish(cmd)

        elif self.state == 'ALIGNING':
            # Align to target yaw
            yaw_error = self.normalize_angle(target['yaw'] - self.robot_yaw)
            
            if abs(yaw_error) < self.YAW_TOLERANCE:
                self.get_logger().info(f'Aligned at {target_id}. Moving to next.')
                self.current_waypoint_idx += 1
                self.state = 'NAVIGATING'
                self.stop_robot()
            else:
                cmd.angular.z = self.KP_ANGULAR * yaw_error
                cmd.angular.z = max(-self.ANGULAR_VEL, min(cmd.angular.z, self.ANGULAR_VEL))
                self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = Task2ANavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
