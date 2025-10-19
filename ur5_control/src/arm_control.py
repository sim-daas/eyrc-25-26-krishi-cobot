#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.spatial.transform import Rotation
import time


class ArmControlNode(Node):
    def __init__(self):
        super().__init__('arm_control_node')
        
        # Publishers
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/delta_joint_cmds', 10)
        
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
        self.control_phase = "PRE_POSITION"
        
        # Tolerances
        self.position_tolerance = 0.15
        self.orientation_tolerance = 0.15  # radians (~8.6 degrees)
        
        # Velocity limits (more aggressive for orientation)
        self.max_linear_velocity_coarse = 0.3
        self.max_angular_velocity_coarse = 0.8  # Increased
        self.max_linear_velocity_fine = 0.12
        self.max_angular_velocity_fine = 0.4    # Increased
        
        # Distance thresholds
        self.far_distance = 0.4
        self.medium_distance = 0.2
        self.close_distance = 0.12
        
        # Control gains (increased orientation gain significantly)
        self.position_gain_base = 2.0
        self.orientation_gain_base = 3.0  # DOUBLED for stronger orientation control
        
        self.timeout = 40.0  # Increased timeout
        self.waypoint_start_time = None
        self.phase_start_time = None
        self.tf_ready = False
        self.last_control_mode = ""
        
        # Pre-positioning parameters
        self.pre_position_duration = 2.0
        self.pre_positioned = False
        
        # Control timer (100 Hz)
        self.control_timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info('Arm Control Node initialized (Enhanced Orientation Control)')
        self.get_logger().info('Waiting for TF and joint states...')
        
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
            
            position = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            
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
        """
        Compute orientation error using scipy Rotation (more robust).
        Handles quaternion double-coverage properly.
        """
        # Normalize quaternions
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        # Use scipy for robust rotation difference
        r_current = Rotation.from_quat(current_quat)
        r_target = Rotation.from_quat(target_quat)
        
        # Compute relative rotation
        r_diff = r_target * r_current.inv()
        
        # Get rotation angle (always positive, 0 to pi)
        angle = r_diff.magnitude()
        
        return angle
    
    def compute_angular_velocity_from_quaternion_error(self, current_quat, target_quat):
        """
        Compute angular velocity using scipy Rotation for robustness.
        Returns angular velocity vector in base_link frame.
        """
        # Normalize
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        # Create Rotation objects
        r_current = Rotation.from_quat(current_quat)
        r_target = Rotation.from_quat(target_quat)
        
        # Compute relative rotation (what rotation gets us from current to target)
        r_error = r_target * r_current.inv()
        
        # Convert to rotation vector (axis * angle)
        rotvec = r_error.as_rotvec()
        
        # This is our angular velocity direction and magnitude
        return rotvec
        
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
    
    def pre_position_base_joint(self, target_pos):
        """Rotate base joint (shoulder_pan) to face target"""
        target_angle = np.arctan2(target_pos[1], target_pos[0])
        current_angle = self.joint_positions[0] if self.joint_positions is not None else 0.0
        
        angle_diff = target_angle - current_angle
        
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Proportional control
        joint_vel = np.zeros(6)
        joint_vel[0] = np.clip(angle_diff * 1.5, -1.0, 1.0)
        
        msg = Float64MultiArray()
        msg.data = joint_vel.tolist()
        self.joint_pub.publish(msg)
        
        return abs(angle_diff) < 0.1
    
    def compute_blending_weights(self, distance_to_target, orientation_error, control_phase):
        """
        Compute position and orientation weights.
        NOW: Give more weight to orientation throughout the motion.
        """
        
        if control_phase == "COARSE_APPROACH":
            if distance_to_target > self.far_distance:
                # Far: Still focus on position, but keep orientation active
                return 0.80, 0.20  # Reduced position dominance
            elif distance_to_target > self.medium_distance:
                return 0.65, 0.35
            else:
                # Getting closer: balance more
                return 0.55, 0.45
        
        else:  # FINE_TUNE phase
            # In fine tune, orientation becomes more important
            if orientation_error > 0.3:  # Large orientation error
                return 0.30, 0.70  # Heavily prioritize orientation
            elif distance_to_target > self.close_distance:
                return 0.45, 0.55  # Slightly favor orientation
            else:
                # Very close: orientation is critical
                return 0.25, 0.75  # Heavily prioritize orientation
        
    def compute_velocity_scaling(self, distance_to_target, orientation_error, control_phase):
        """Compute velocity scaling - don't slow down too much for orientation errors"""
        
        if control_phase == "COARSE_APPROACH":
            if distance_to_target < self.medium_distance:
                distance_scale = 0.6 + 0.4 * (distance_to_target / self.medium_distance)
            else:
                distance_scale = 1.0
        else:  # FINE_TUNE
            if distance_to_target < self.close_distance:
                distance_scale = 0.5 + 0.5 * (distance_to_target / self.close_distance)
            else:
                distance_scale = 0.8
        
        # Don't slow down as much for orientation errors
        if orientation_error > 1.0:  # Very large error (>57 degrees)
            orientation_scale = 0.8  # Keep moving
        elif orientation_error > 0.5:
            orientation_scale = 0.9
        else:
            orientation_scale = 1.0
        
        return min(distance_scale, orientation_scale, 1.0)
        
    def compute_twist_command(self, current_pos, current_quat, target_pos, target_quat):
        """Compute velocity command with IMPROVED orientation control"""
        
        # Compute errors
        pos_error_vector = target_pos - current_pos
        distance_to_target = np.linalg.norm(pos_error_vector)
        orientation_error = self.compute_orientation_error(current_quat, target_quat)
        
        # Get blending weights (now considers orientation error too)
        pos_weight, ori_weight = self.compute_blending_weights(
            distance_to_target, orientation_error, self.control_phase
        )
        
        # Log control mode change
        control_mode = f"{self.control_phase} - pos:{pos_weight:.2f}/ori:{ori_weight:.2f}"
        if control_mode != self.last_control_mode:
            self.get_logger().info(
                f'Control mode: {control_mode} '
                f'(dist: {distance_to_target:.3f}m, ori_err: {np.degrees(orientation_error):.1f}°)'
            )
            self.last_control_mode = control_mode
        
        # Position control (proportional)
        linear_vel = pos_error_vector * self.position_gain_base * pos_weight
        
        # Orientation control (using scipy rotation vector - MUCH MORE ROBUST)
        angular_vel = self.compute_angular_velocity_from_quaternion_error(
            current_quat, target_quat
        ) * self.orientation_gain_base * ori_weight
        
        # Apply velocity scaling
        velocity_scale = self.compute_velocity_scaling(
            distance_to_target, orientation_error, self.control_phase
        )
        
        linear_vel *= velocity_scale
        angular_vel *= velocity_scale
        
        # Apply velocity limits based on phase
        if self.control_phase == "COARSE_APPROACH":
            max_linear = self.max_linear_velocity_coarse
            max_angular = self.max_angular_velocity_coarse
        else:
            max_linear = self.max_linear_velocity_fine
            max_angular = self.max_angular_velocity_fine
        
        linear_speed = np.linalg.norm(linear_vel)
        if linear_speed > max_linear:
            linear_vel = linear_vel * (max_linear / linear_speed)
        
        angular_speed = np.linalg.norm(angular_vel)
        if angular_speed > max_angular:
            angular_vel = angular_vel * (max_angular / angular_speed)
        
        return linear_vel, angular_vel
        
    def check_waypoint_reached(self, current_pos, current_quat, target_pos, target_quat):
        """Check if BOTH position AND orientation are within tolerance"""
        pos_error = self.compute_position_error(current_pos, target_pos)
        ori_error = self.compute_orientation_error(current_quat, target_quat)
        
        position_ok = pos_error < self.position_tolerance
        orientation_ok = ori_error < self.orientation_tolerance
        
        return position_ok and orientation_ok
        
    def control_loop(self):
        """Main control loop with phases"""
        if self.joint_positions is None:
            return
            
        # Check if all waypoints completed
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info('✓ All waypoints reached! Shutting down...')
            self.stop_robot()
            time.sleep(1.0)
            rclpy.shutdown()
            return
        
        current_pos, current_quat = self.get_current_pose()
        if current_pos is None:
            return
        
        target_pos, target_quat = self.waypoints[self.current_waypoint_idx]
        
        # Initialize timer for this waypoint
        if self.waypoint_start_time is None:
            self.waypoint_start_time = time.time()
            self.phase_start_time = time.time()
            self.control_phase = "PRE_POSITION"
            self.pre_positioned = False
            
            # Log initial orientation error for debugging
            initial_ori_error = self.compute_orientation_error(current_quat, target_quat)
            self.get_logger().info(
                f'\n{"="*60}\n'
                f'→ Moving to Waypoint {self.current_waypoint_idx + 1}/3\n'
                f'  Target Position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]\n'
                f'  Target Orientation: [{target_quat[0]:.3f}, {target_quat[1]:.3f}, {target_quat[2]:.3f}, {target_quat[3]:.3f}]\n'
                f'  Initial Orientation Error: {np.degrees(initial_ori_error):.2f}°\n'
                f'  Phase: PRE_POSITION (rotating base joint)\n'
                f'{"="*60}'
            )
        
        elapsed_time = time.time() - self.waypoint_start_time
        if elapsed_time > self.timeout:
            pos_error = self.compute_position_error(current_pos, target_pos)
            ori_error = self.compute_orientation_error(current_quat, target_quat)
            self.get_logger().warn(
                f'⚠ Timeout ({self.timeout}s) reached for waypoint {self.current_waypoint_idx + 1}.\n'
                f'  Final errors - Pos: {pos_error:.4f}m, Ori: {ori_error:.4f}rad ({np.degrees(ori_error):.2f}°)\n'
                f'  Moving to next waypoint...'
            )
            self.move_to_next_waypoint()
            return
        
        # Phase 1: PRE_POSITION
        if self.control_phase == "PRE_POSITION":
            phase_elapsed = time.time() - self.phase_start_time
            is_aligned = self.pre_position_base_joint(target_pos)
            
            if is_aligned or phase_elapsed > self.pre_position_duration:
                self.control_phase = "COARSE_APPROACH"
                self.phase_start_time = time.time()
                self.get_logger().info('✓ Pre-positioning complete. Switching to COARSE_APPROACH')
            return
        
        # Compute current errors
        pos_error = self.compute_position_error(current_pos, target_pos)
        ori_error = self.compute_orientation_error(current_quat, target_quat)
        distance_to_target = pos_error
        
        # Phase 2: COARSE_APPROACH
        if self.control_phase == "COARSE_APPROACH":
            if distance_to_target < self.close_distance:
                self.control_phase = "FINE_TUNE"
                self.phase_start_time = time.time()
                self.get_logger().info(
                    f'✓ Close to target ({distance_to_target:.3f}m, ori_err: {np.degrees(ori_error):.1f}°). '
                    f'Switching to FINE_TUNE'
                )
        
        # Check if waypoint reached
        if self.check_waypoint_reached(current_pos, current_quat, target_pos, target_quat):
            self.get_logger().info(
                f'\n{"="*60}\n'
                f'✓ Waypoint {self.current_waypoint_idx + 1}/3 REACHED!\n'
                f'  Position error: {pos_error:.4f}m (tolerance: {self.position_tolerance}m) ✓\n'
                f'  Orientation error: {ori_error:.4f}rad = {np.degrees(ori_error):.2f}° '
                f'(tolerance: {self.orientation_tolerance}rad = {np.degrees(self.orientation_tolerance):.2f}°) ✓\n'
                f'  Time taken: {elapsed_time:.2f}s\n'
                f'  Final phase: {self.control_phase}\n'
                f'{"="*60}'
            )
            
            self.stop_robot()
            self.get_logger().info('Pausing for 1 second...')
            time.sleep(1.0)
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
            self.get_logger().info(
                f'WP{self.current_waypoint_idx + 1} [{elapsed_time:.1f}s] {self.control_phase} - '
                f'Pos: {pos_error:.4f}m {"✓" if pos_error < self.position_tolerance else "✗"}, '
                f'Ori: {ori_error:.4f}rad ({np.degrees(ori_error):.1f}°) {"✓" if ori_error < self.orientation_tolerance else "✗"}'
            )
    
    def move_to_next_waypoint(self):
        """Move to the next waypoint"""
        self.current_waypoint_idx += 1
        self.waypoint_start_time = None
        self.phase_start_time = None
        self.control_phase = "PRE_POSITION"
        self.pre_positioned = False
        self.last_control_mode = ""
        
    def stop_robot(self):
        """Stop the robot by publishing zero velocity"""
        twist_msg = Twist()
        for _ in range(10):
            self.twist_pub.publish(twist_msg)
            time.sleep(0.01)
        
        joint_msg = Float64MultiArray()
        joint_msg.data = [0.0] * 6
        for _ in range(10):
            self.joint_pub.publish(joint_msg)
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
