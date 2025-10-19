#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.spatial.transform import Rotation
import time


class ArmControlNode(Node):
    def __init__(self):
        super().__init__('arm_control_node')
        
        # Publishers
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        # TF Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Waypoints (position [x,y,z], quaternion [x,y,z,w])
        self.waypoints = [
            (np.array([-0.214, -0.532, 0.557]), np.array([0.707, 0.028, 0.034, 0.707])),  # P1
            (np.array([-0.159, 0.501, 0.415]), np.array([0.029, 0.997, 0.045, 0.033])),   # P2
            (np.array([-0.806, 0.010, 0.182]), np.array([-0.684, 0.726, 0.05, 0.008]))    # P3
        ]
        
        # State variables
        self.joint_positions = None
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        self.position_tolerance = 0.15
        self.orientation_tolerance = 0.15  # radians
        self.max_linear_velocity = 0.3  # m/s
        self.max_angular_velocity = 0.5  # rad/s
        self.timeout = 20.0  # seconds per waypoint
        self.waypoint_start_time = None
        self.tf_ready = False
        
        # Control timer (100 Hz)
        self.control_timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info('Arm Control Node initialized. Waiting for TF and joint states...')
        
    def joint_callback(self, msg):
        """Update current joint positions"""
        self.joint_positions = np.array(msg.position[:6])
        
    def get_current_pose(self):
        """Get current end-effector pose from TF"""
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                'base_link', 
                'tool0', 
                rclpy.time.Time()
            )
            
            # Extract position
            position = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            
            # Extract orientation as quaternion [x, y, z, w]
            quaternion = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            
            if not self.tf_ready:
                self.tf_ready = True
                self.get_logger().info('TF transform ready')
            
            return position, quaternion
            
        except Exception as e:
            if self.tf_ready:
                self.get_logger().warn(f'Failed to get transform: {str(e)}')
            return None, None
        
    def compute_position_error(self, current_pos, target_pos):
        """Compute position error"""
        return np.linalg.norm(target_pos - current_pos)
        
    def compute_orientation_error(self, current_quat, target_quat):
        """Compute orientation error in radians"""
        # Normalize quaternions
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        # Compute quaternion difference
        q_diff = self.quaternion_multiply(self.quaternion_inverse(current_quat), target_quat)
        
        # Extract rotation angle
        angle = 2 * np.arccos(np.clip(abs(q_diff[3]), 0, 1))
        return angle
        
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [x, y, z, w]"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
        
    def quaternion_inverse(self, q):
        """Compute quaternion inverse [x, y, z, w]"""
        return np.array([-q[0], -q[1], -q[2], q[3]])
        
    def compute_twist_command(self, current_pos, current_quat, target_pos, target_quat):
        """Compute velocity command to reach target"""
        # Position error (proportional control)
        pos_error = target_pos - current_pos
        linear_vel = pos_error * 2.0  # Proportional gain
        
        # Limit linear velocity
        linear_speed = np.linalg.norm(linear_vel)
        if linear_speed > self.max_linear_velocity:
            linear_vel = linear_vel * (self.max_linear_velocity / linear_speed)
        
        # Orientation error (axis-angle representation)
        current_quat_norm = current_quat / np.linalg.norm(current_quat)
        target_quat_norm = target_quat / np.linalg.norm(target_quat)
        
        q_diff = self.quaternion_multiply(self.quaternion_inverse(current_quat_norm), target_quat_norm)
        
        # Convert to axis-angle
        if abs(q_diff[3]) < 1.0:
            angle = 2 * np.arccos(np.clip(q_diff[3], -1.0, 1.0))
            if angle > 1e-6:
                axis = q_diff[:3] / np.sin(angle / 2)
                angular_vel = axis * angle * 1.5  # Proportional gain
            else:
                angular_vel = np.zeros(3)
        else:
            angular_vel = np.zeros(3)
        
        # Limit angular velocity
        angular_speed = np.linalg.norm(angular_vel)
        if angular_speed > self.max_angular_velocity:
            angular_vel = angular_vel * (self.max_angular_velocity / angular_speed)
        
        return linear_vel, angular_vel
        
    def check_waypoint_reached(self, current_pos, current_quat, target_pos, target_quat):
        """Check if waypoint is reached within tolerance"""
        pos_error = self.compute_position_error(current_pos, target_pos)
        ori_error = self.compute_orientation_error(current_quat, target_quat)
        
        return pos_error < self.position_tolerance and ori_error < self.orientation_tolerance
        
    def control_loop(self):
        """Main control loop"""
        if self.joint_positions is None:
            return
            
        # Check if all waypoints completed
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints reached! Shutting down...')
            self.stop_robot()
            time.sleep(1.0)
            rclpy.shutdown()
            return
        
        # Get current pose
        current_pos, current_quat = self.get_current_pose()
        if current_pos is None:
            return
        
        # Get target waypoint
        target_pos, target_quat = self.waypoints[self.current_waypoint_idx]
        
        # Initialize timer for this waypoint
        if self.waypoint_start_time is None:
            self.waypoint_start_time = time.time()
            self.get_logger().info(
                f'Moving to waypoint {self.current_waypoint_idx + 1}: '
                f'Position {target_pos}, Orientation {target_quat}'
            )
        
        # Check timeout
        elapsed_time = time.time() - self.waypoint_start_time
        if elapsed_time > self.timeout:
            self.get_logger().warn(
                f'Timeout reached for waypoint {self.current_waypoint_idx + 1}. '
                f'Moving to next waypoint...'
            )
            self.move_to_next_waypoint()
            return
        
        # Check if waypoint reached
        if self.check_waypoint_reached(current_pos, current_quat, target_pos, target_quat):
            pos_error = self.compute_position_error(current_pos, target_pos)
            ori_error = self.compute_orientation_error(current_quat, target_quat)
            
            self.get_logger().info(
                f'Waypoint {self.current_waypoint_idx + 1} reached! '
                f'Position error: {pos_error:.4f}m, Orientation error: {ori_error:.4f}rad'
            )
            
            # Stop and wait
            self.stop_robot()
            time.sleep(1.0)
            
            # Move to next waypoint
            self.move_to_next_waypoint()
            return
        
        # Compute and publish velocity command
        linear_vel, angular_vel = self.compute_twist_command(
            current_pos, current_quat, target_pos, target_quat
        )
        
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_vel[0])
        twist_msg.linear.y = float(linear_vel[1])
        twist_msg.linear.z = float(linear_vel[2])
        twist_msg.angular.x = float(angular_vel[0])
        twist_msg.angular.y = float(angular_vel[1])
        twist_msg.angular.z = float(angular_vel[2])
        
        self.twist_pub.publish(twist_msg)
        
        # Log progress every 2 seconds
        if int(elapsed_time * 10) % 20 == 0:
            pos_error = self.compute_position_error(current_pos, target_pos)
            ori_error = self.compute_orientation_error(current_quat, target_quat)
            self.get_logger().info(
                f'Waypoint {self.current_waypoint_idx + 1} - '
                f'Position error: {pos_error:.4f}m, Orientation error: {ori_error:.4f}rad'
            )
    
    def move_to_next_waypoint(self):
        """Move to the next waypoint"""
        self.current_waypoint_idx += 1
        self.waypoint_start_time = None
        
    def stop_robot(self):
        """Stop the robot by publishing zero velocity"""
        twist_msg = Twist()
        for _ in range(10):  # Send multiple stop commands
            self.twist_pub.publish(twist_msg)
            time.sleep(0.01)


def main(args=None):
    rclpy.init(args=args)
    node = ArmControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
