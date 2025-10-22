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
        
        # SAFE TRANSIT JOINT CONFIGURATION
        self.safe_transit_joints = np.array([
            -3.973,  # shoulder_pan_joint
            -1.596,  # shoulder_lift_joint
             0.0,    # elbow_joint
             0.0,    # wrist_1_joint
             0.0,    # wrist_2_joint
             0.0     # wrist_3_joint
        ])
        
        # Target waypoints - Cartesian poses
        self.target_waypoints = [
            (np.array([-0.214, -0.532, 0.557]), np.array([0.707, 0.028, 0.034, 0.707])),  # P1
            (np.array([-0.159, 0.501, 0.415]), np.array([0.029, 0.997, 0.045, 0.033])),   # P2
            (np.array([-0.806, 0.010, 0.182]), np.array([-0.684, 0.726, 0.05, 0.008]))    # P3
        ]
        
        # Build full waypoint sequence
        self.waypoints = self.build_waypoint_sequence()
        
        # State variables
        self.joint_positions = None
        self.current_waypoint_idx = 0
        self.control_mode = "JOINT_CONTROL"
        
        # Tolerances - LOOSENED FOR TRANSITS
        self.position_tolerance = 0.15
        self.orientation_tolerance = 0.15
        self.joint_tolerance = 1.0  # INCREASED: ~8.6° (was 2.86°) - faster transit
        
        # Velocity limits - INCREASED FOR TRANSITS
        self.max_joint_velocity = 1.5  # INCREASED: rad/s for faster transit
        self.max_linear_velocity_coarse = 0.5
        self.max_angular_velocity_coarse = 1.0
        self.max_linear_velocity_fine = 0.19
        self.max_angular_velocity_fine = 0.7
        
        # Distance thresholds for Cartesian control
        self.far_distance = 0.4
        self.medium_distance = 0.2
        self.close_distance = 0.12
        
        # Control gains
        self.joint_gain = 4  # INCREASED: P gain for faster joint control
        self.position_gain_base = 2.5
        self.orientation_gain_base = 3.5
        
        self.timeout = 40.0
        self.waypoint_start_time = None
        self.tf_ready = False
        self.last_log_time = 0
        
        # Cartesian servo phase
        self.servo_phase = "COARSE_APPROACH"
        
        # Control timer (100 Hz)
        self.control_timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info('Arm Control Node initialized (Fast Transit + Precise Target)')
        self.get_logger().info(f'Total waypoints: {len(self.waypoints)}')
        self.get_logger().info('Sequence: P1 → Transit → P2 → P3 (optimized path)')
        self.get_logger().info(f'Transit tolerance: {np.degrees(self.joint_tolerance):.1f}° (loose)')
        self.get_logger().info(f'Target tolerance: {self.position_tolerance}m, {np.degrees(self.orientation_tolerance):.1f}°')
        self.get_logger().info('Waiting for joint states...')
    
    def build_waypoint_sequence(self):
        """
        Build optimized waypoint sequence.
        Sequence: P1 → Transit (only between P1 and P2) → P2 → P3
        
        Transit only needed between P1 and P2 to avoid obstacles.
        P2 to P3 can be direct (no obstacles in that path).
        """
        sequence = []
        
        # Waypoint 1: P1 (direct from start)
        sequence.append({
            'type': 'cartesian',
            'target_joints': None,
            'target_pose': self.target_waypoints[0],
            'description': 'Target P1 (Cartesian)',
            'is_target': True
        })
        
        # Transit point (only after P1, before P2)
        sequence.append({
            'type': 'joint',
            'target_joints': self.safe_transit_joints.copy(),
            'target_pose': None,
            'description': 'Transit (joint) between P1 and P2',
            'is_target': False
        })
        
        # Waypoint 2: P2
        sequence.append({
            'type': 'cartesian',
            'target_joints': None,
            'target_pose': self.target_waypoints[1],
            'description': 'Target P2 (Cartesian)',
            'is_target': True
        })
        
        # Waypoint 3: P3 (direct from P2 - no transit needed)
        sequence.append({
            'type': 'cartesian',
            'target_joints': None,
            'target_pose': self.target_waypoints[2],
            'description': 'Target P3 (Cartesian)',
            'is_target': True
        })
        
        return sequence
    
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
            return None, None
    
    # ========== JOINT SPACE CONTROL ==========
    
    def compute_joint_velocities(self, target_joints):
        """Compute joint velocities using proportional control"""
        if self.joint_positions is None:
            return np.zeros(6)
        
        joint_errors = target_joints - self.joint_positions
        joint_velocities = joint_errors * self.joint_gain
        
        # Limit velocities
        for i in range(6):
            joint_velocities[i] = np.clip(
                joint_velocities[i], 
                -self.max_joint_velocity, 
                self.max_joint_velocity
            )
        
        return joint_velocities
    
    def check_joint_waypoint_reached(self, target_joints):
        """Check if joint-space waypoint is reached"""
        if self.joint_positions is None:
            return False
        
        joint_errors = np.abs(target_joints - self.joint_positions)
        max_error = np.max(joint_errors)
        
        return max_error < self.joint_tolerance
    
    # ========== CARTESIAN SPACE CONTROL ==========
    
    def compute_position_error(self, current_pos, target_pos):
        """Compute position error"""
        return np.linalg.norm(target_pos - current_pos)
        
    def compute_orientation_error(self, current_quat, target_quat):
        """Compute orientation error using scipy Rotation"""
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        r_current = Rotation.from_quat(current_quat)
        r_target = Rotation.from_quat(target_quat)
        r_diff = r_target * r_current.inv()
        
        return r_diff.magnitude()
    
    def compute_angular_velocity_from_quaternion_error(self, current_quat, target_quat):
        """Compute angular velocity using scipy Rotation"""
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        r_current = Rotation.from_quat(current_quat)
        r_target = Rotation.from_quat(target_quat)
        r_error = r_target * r_current.inv()
        
        return r_error.as_rotvec()
    
    def compute_blending_weights(self, distance_to_target, orientation_error):
        """Compute position and orientation weights based on distance"""
        if self.servo_phase == "COARSE_APPROACH":
            if distance_to_target > self.far_distance:
                return 0.80, 0.20
            elif distance_to_target > self.medium_distance:
                return 0.65, 0.35
            else:
                return 0.55, 0.45
        else:  # FINE_TUNE
            if orientation_error > 0.3:
                return 0.30, 0.70
            elif distance_to_target > self.close_distance:
                return 0.45, 0.55
            else:
                return 0.25, 0.75
        
    def compute_velocity_scaling(self, distance_to_target, orientation_error):
        """Compute velocity scaling for smooth motion"""
        if self.servo_phase == "COARSE_APPROACH":
            if distance_to_target < self.medium_distance:
                distance_scale = 0.6 + 0.4 * (distance_to_target / self.medium_distance)
            else:
                distance_scale = 1.0
        else:  # FINE_TUNE
            if distance_to_target < self.close_distance:
                distance_scale = 0.5 + 0.5 * (distance_to_target / self.close_distance)
            else:
                distance_scale = 0.8
        
        if orientation_error > 1.0:
            orientation_scale = 0.8
        elif orientation_error > 0.5:
            orientation_scale = 0.9
        else:
            orientation_scale = 1.0
        
        return min(distance_scale, orientation_scale, 1.0)
        
    def compute_twist_command(self, current_pos, current_quat, target_pos, target_quat):
        """Compute Cartesian velocity command"""
        pos_error_vector = target_pos - current_pos
        distance_to_target = np.linalg.norm(pos_error_vector)
        orientation_error = self.compute_orientation_error(current_quat, target_quat)
        
        pos_weight, ori_weight = self.compute_blending_weights(
            distance_to_target, orientation_error
        )
        
        linear_vel = pos_error_vector * self.position_gain_base * pos_weight
        angular_vel = self.compute_angular_velocity_from_quaternion_error(
            current_quat, target_quat
        ) * self.orientation_gain_base * ori_weight
        
        velocity_scale = self.compute_velocity_scaling(distance_to_target, orientation_error)
        linear_vel *= velocity_scale
        angular_vel *= velocity_scale
        
        if self.servo_phase == "COARSE_APPROACH":
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
        
        return linear_vel, angular_vel, distance_to_target, orientation_error
        
    def check_cartesian_waypoint_reached(self, current_pos, current_quat, target_pos, target_quat):
        """Check if Cartesian waypoint is reached"""
        pos_error = self.compute_position_error(current_pos, target_pos)
        ori_error = self.compute_orientation_error(current_quat, target_quat)
        
        position_ok = pos_error < self.position_tolerance
        orientation_ok = ori_error < self.orientation_tolerance
        
        return position_ok and orientation_ok, pos_error, ori_error
    
    # ========== MAIN CONTROL LOOP ==========
    
    def control_loop(self):
        """Main control loop - hybrid joint/Cartesian control"""
        if self.joint_positions is None:
            return
        
        # Check if all waypoints completed
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info('✓ All waypoints reached! Shutting down...')
            self.stop_robot()
            time.sleep(0.5)  # Brief pause before shutdown
            rclpy.shutdown()
            return
        
        # Get current waypoint
        waypoint = self.waypoints[self.current_waypoint_idx]
        
        # Initialize timer for this waypoint
        if self.waypoint_start_time is None:
            self.waypoint_start_time = time.time()
            self.last_log_time = time.time()
            
            if waypoint['type'] == 'joint':
                self.control_mode = "JOINT_CONTROL"
            else:
                self.control_mode = "CARTESIAN_SERVO"
                self.servo_phase = "COARSE_APPROACH"
            
            self.get_logger().info(
                f'\n{"="*60}\n'
                f'→ Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}: {waypoint["description"]}\n'
                f'  Control Mode: {self.control_mode}\n'
                f'{"="*60}'
            )
        
        # Check timeout
        elapsed_time = time.time() - self.waypoint_start_time
        if elapsed_time > self.timeout:
            self.get_logger().warn(
                f'⚠ Timeout ({self.timeout}s) for {waypoint["description"]}. Moving to next...'
            )
            self.move_to_next_waypoint()
            return
        
        # ========== JOINT SPACE CONTROL (FAST TRANSITS) ==========
        if self.control_mode == "JOINT_CONTROL":
            target_joints = waypoint['target_joints']
            
            # Compute and publish joint velocities
            joint_velocities = self.compute_joint_velocities(target_joints)
            
            msg = Float64MultiArray()
            msg.data = joint_velocities.tolist()
            self.joint_pub.publish(msg)
            
            # Check if reached (with loose tolerance)
            if self.check_joint_waypoint_reached(target_joints):
                joint_errors = np.abs(target_joints - self.joint_positions)
                self.get_logger().info(
                    f'\n{"="*60}\n'
                    f'✓ {waypoint["description"]} REACHED (fast transit)!\n'
                    f'  Max joint error: {np.max(joint_errors):.4f} rad ({np.degrees(np.max(joint_errors)):.2f}°)\n'
                    f'  Time taken: {elapsed_time:.2f}s\n'
                    f'{"="*60}'
                )
                
                self.stop_robot()
                time.sleep(0.1)  # REDUCED: Just 0.1s for stability
                self.move_to_next_waypoint()
                return
            
            # Log progress every 2 seconds
            if time.time() - self.last_log_time > 2.0:
                joint_errors = np.abs(target_joints - self.joint_positions)
                self.get_logger().info(
                    f'[{elapsed_time:.1f}s] JOINT_CONTROL (fast) - Max error: {np.max(joint_errors):.4f} rad'
                )
                self.last_log_time = time.time()
        
        # ========== CARTESIAN SPACE CONTROL (PRECISE TARGETS) ==========
        elif self.control_mode == "CARTESIAN_SERVO":
            # Get current pose
            current_pos, current_quat = self.get_current_pose()
            if current_pos is None:
                return
            
            target_pos, target_quat = waypoint['target_pose']
            
            # Phase transition: COARSE → FINE
            pos_error = self.compute_position_error(current_pos, target_pos)
            if self.servo_phase == "COARSE_APPROACH" and pos_error < self.close_distance:
                self.servo_phase = "FINE_TUNE"
                self.get_logger().info('✓ Switching to FINE_TUNE phase')
            
            # Compute and publish twist command
            linear_vel, angular_vel, dist, ori_err = self.compute_twist_command(
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
            
            # Check if reached
            reached, pos_err, ori_err = self.check_cartesian_waypoint_reached(
                current_pos, current_quat, target_pos, target_quat
            )
            
            if reached:
                self.get_logger().info(
                    f'\n{"="*60}\n'
                    f'✓ {waypoint["description"]} REACHED!\n'
                    f'  Position error: {pos_err:.4f}m ✓\n'
                    f'  Orientation error: {ori_err:.4f}rad = {np.degrees(ori_err):.2f}° ✓\n'
                    f'  Time taken: {elapsed_time:.2f}s\n'
                    f'{"="*60}'
                )
                
                self.stop_robot()
                
                # NO PAUSE - immediately move to next (removed 1 second delay)
                self.move_to_next_waypoint()
                return
            
            # Log progress every 2 seconds
            if time.time() - self.last_log_time > 2.0:
                self.get_logger().info(
                    f'[{elapsed_time:.1f}s] {self.servo_phase} - '
                    f'Pos: {pos_err:.4f}m, Ori: {np.degrees(ori_err):.1f}°'
                )
                self.last_log_time = time.time()
    
    def move_to_next_waypoint(self):
        """Move to next waypoint"""
        self.current_waypoint_idx += 1
        self.waypoint_start_time = None
        
    def stop_robot(self):
        """Stop both joint and Cartesian control"""
        # Stop Cartesian servo
        twist_msg = Twist()
        for _ in range(5):  # REDUCED: 5 iterations instead of 10
            self.twist_pub.publish(twist_msg)
            time.sleep(0.005)  # REDUCED: 5ms instead of 10ms
        
        # Stop joint control
        joint_msg = Float64MultiArray()
        joint_msg.data = [0.0] * 6
        for _ in range(5):  # REDUCED: 5 iterations instead of 10
            self.joint_pub.publish(joint_msg)
            time.sleep(0.005)  # REDUCED: 5ms instead of 10ms


def main(args=None):
    time.sleep(0.3)
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
