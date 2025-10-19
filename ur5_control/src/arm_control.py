#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import numpy as np
from scipy.spatial.transform import Rotation
import time

class UR5WaypointServo(Node):
    def __init__(self):
        super().__init__('ur5_waypoint_servo')
        
        # Publisher for velocity commands
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        # TF2 for getting current end-effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Original waypoints - these should all be reachable
        self.waypoints = [
            {
                'name': 'P1',
                'position': np.array([-0.214, -0.532, 0.557]),
                'orientation': np.array([0.707, 0.028, 0.034, 0.707])  # x,y,z,w
            },
            {
                'name': 'P2',
                'position': np.array([-0.159, 0.501, 0.415]),
                'orientation': np.array([0.029, 0.997, 0.045, 0.033])
            },
            {
                'name': 'P3',
                'position': np.array([-0.806, 0.010, 0.182]),
                'orientation': np.array([-0.684, 0.726, 0.05, 0.008])
            }
        ]
        
        # Control parameters
        self.position_tolerance = 0.15
        self.orientation_tolerance = 0.15
        self.max_retries = 1  # Only 1 attempt per waypoint (15s total)
        self.stop_duration = 1.0
        self.waypoint_timeout = 15.0  # 15 seconds per waypoint
        
        # Conservative servo gains for safety
        self.k_linear = 0.25
        self.k_angular = 0.4
        self.max_linear_vel = 0.12
        self.max_angular_vel = 0.25
        
        # CORRECTED Safety limits for UR5
        # UR5 specifications: max reach = 850mm from base
        self.max_workspace_radius = 0.85  # UR5's actual maximum reach
        self.warning_radius = 0.80  # Start warning at 800mm
        self.min_z_height = 0.05  # Minimum height above base
        
        self.get_logger().info('UR5 Waypoint Servo Controller initialized')
        self.get_logger().info('Waiting for TF frames to become available...')
        
        # Verify waypoints are within reach
        self.verify_waypoints()

    def verify_waypoints(self):
        """Check if waypoints are within UR5's workspace"""
        self.get_logger().info('='*60)
        self.get_logger().info('WAYPOINT VALIDATION (UR5 max reach: 850mm)')
        self.get_logger().info('='*60)
        
        all_valid = True
        
        for wp in self.waypoints:
            pos = wp['position']
            distance_xy = np.sqrt(pos[0]**2 + pos[1]**2)
            distance_3d = np.linalg.norm(pos)
            
            # Check constraints
            is_valid = True
            issues = []
            warnings = []
            
            if distance_3d > self.max_workspace_radius:
                is_valid = False
                issues.append(f"3D distance {distance_3d:.3f}m exceeds UR5 max reach {self.max_workspace_radius}m")
            elif distance_3d > self.warning_radius:
                warnings.append(f"Near workspace limit (distance: {distance_3d:.3f}m)")
            
            if pos[2] < self.min_z_height:
                is_valid = False
                issues.append(f"Z height {pos[2]:.3f}m below minimum {self.min_z_height}m")
            
            # XY limit is less restrictive with proper Z height
            if distance_xy > 0.82:  # Very close to absolute limit
                warnings.append(f"XY distance {distance_xy:.3f}m near maximum")
            
            # Log results
            status = "✓ VALID" if is_valid else "✗ INVALID"
            color = "\033[92m" if is_valid else "\033[91m"
            reset = "\033[0m"
            
            self.get_logger().info(
                f"{color}{status}{reset} {wp['name']}: "
                f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
            )
            self.get_logger().info(
                f"       XY: {distance_xy:.3f}m, 3D: {distance_3d:.3f}m"
            )
            
            if issues:
                for issue in issues:
                    self.get_logger().error(f"       ✗ {issue}")
                all_valid = False
            
            if warnings:
                for warning in warnings:
                    self.get_logger().warn(f"       ⚠️  {warning}")
        
        self.get_logger().info('='*60)
        
        if not all_valid:
            self.get_logger().error(
                "✗ Some waypoints are outside UR5 workspace! "
                "Motion will likely fail."
            )
        else:
            self.get_logger().info("✓ All waypoints validated successfully")

    def wait_for_tf_frames(self, timeout=30.0):
        """Wait for required TF frames to become available"""
        start_time = time.time()
        base_frame = 'base_link'
        tool_frame = 'tool0'
        
        while time.time() - start_time < timeout:
            try:
                self.tf_buffer.lookup_transform(
                    base_frame,
                    tool_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                self.get_logger().info(f'✓ TF frames are available: {base_frame} -> {tool_frame}')
                
                # Log current pose
                current_pos, current_ori = self.get_current_pose()
                if current_pos is not None:
                    self.get_logger().info(
                        f'Current tool0 position: '
                        f'[{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]'
                    )
                
                return True
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().info(f'Waiting for TF... ({int(time.time() - start_time)}s elapsed)')
                time.sleep(1.0)
                rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().error(f'Timeout waiting for TF frames after {timeout}s')
        return False

    def get_current_pose(self):
        """Get current end-effector pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'tool0',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            orientation = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            return position, orientation
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            return None, None

    def quaternion_error(self, q_current, q_target):
        """Calculate orientation error as axis-angle"""
        q_current = q_current / np.linalg.norm(q_current)
        q_target = q_target / np.linalg.norm(q_target)
        
        r_current = Rotation.from_quat(q_current)
        r_target = Rotation.from_quat(q_target)
        r_error = r_target * r_current.inv()
        
        rotvec = r_error.as_rotvec()
        return rotvec

    def compute_servo_command(self, current_pos, current_ori, target_pos, target_ori):
        """Compute velocity command using proportional control"""
        twist = Twist()
        
        # Check if we're approaching workspace limits
        current_distance = np.linalg.norm(current_pos)
        
        # Reduce velocity as we approach the warning threshold
        safety_factor = 1.0
        if current_distance > self.warning_radius:  # Approaching 850mm limit
            # Gradually reduce speed from 800mm to 850mm
            safety_factor = max(0.2, 1.0 - (current_distance - self.warning_radius) / 
                              (self.max_workspace_radius - self.warning_radius))
        
        # Linear velocity (proportional to position error)
        pos_error = target_pos - current_pos
        linear_vel = self.k_linear * pos_error * safety_factor
        
        # Clamp linear velocity
        linear_speed = np.linalg.norm(linear_vel)
        if linear_speed > self.max_linear_vel:
            linear_vel = linear_vel / linear_speed * self.max_linear_vel
        
        twist.linear.x = float(linear_vel[0])
        twist.linear.y = float(linear_vel[1])
        twist.linear.z = float(linear_vel[2])
        
        # Angular velocity (proportional to orientation error)
        ori_error = self.quaternion_error(current_ori, target_ori)
        angular_vel = self.k_angular * ori_error * safety_factor
        
        # Clamp angular velocity
        angular_speed = np.linalg.norm(angular_vel)
        if angular_speed > self.max_angular_vel:
            angular_vel = angular_vel / angular_speed * self.max_angular_vel
        
        twist.angular.x = float(angular_vel[0])
        twist.angular.y = float(angular_vel[1])
        twist.angular.z = float(angular_vel[2])
        
        return twist

    def check_waypoint_reached(self, current_pos, current_ori, target_pos, target_ori):
        """Check if waypoint is reached within tolerance"""
        pos_error = np.linalg.norm(target_pos - current_pos)
        ori_error_vec = self.quaternion_error(current_ori, target_ori)
        ori_error = np.linalg.norm(ori_error_vec)
        
        position_ok = pos_error < self.position_tolerance
        orientation_ok = ori_error < self.orientation_tolerance
        
        return position_ok and orientation_ok, pos_error, ori_error

    def stop_motion(self):
        """Send zero velocity command"""
        twist = Twist()
        for _ in range(5):  # Send multiple stop commands
            self.twist_pub.publish(twist)
            time.sleep(0.01)

    def move_to_waypoint(self, waypoint, timeout=None):
        """Move to a single waypoint with 15 second timeout"""
        if timeout is None:
            timeout = self.waypoint_timeout
        
        target_pos = waypoint['position']
        target_ori = waypoint['orientation']
        name = waypoint['name']
        
        for attempt in range(self.max_retries):
            self.get_logger().info(
                f'Moving to {name} - {timeout:.0f}s timeout '
                f'(Attempt {attempt + 1}/{self.max_retries})'
            )
            
            start_time = time.time()
            # Use simple time-based loop instead of rate
            loop_dt = 0.02  # 50 Hz (20ms per loop)
            
            consecutive_failures = 0
            max_consecutive_failures = 5
            last_pos_error = float('inf')
            stuck_count = 0
            last_log_time = start_time
            min_error_seen = float('inf')  # Track minimum error achieved
            
            while True:
                loop_start = time.time()
                elapsed = loop_start - start_time
                
                # Check timeout first
                if elapsed >= timeout:
                    self.get_logger().warn(
                        f'⏱️  Timeout ({timeout:.0f}s) reached for {name} - Moving to next waypoint'
                    )
                    break
                
                # Get current pose
                current_pos, current_ori = self.get_current_pose()
                if current_pos is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        self.get_logger().error(f'Too many consecutive TF lookup failures')
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                
                # Check if reached
                reached, pos_err, ori_err = self.check_waypoint_reached(
                    current_pos, current_ori, target_pos, target_ori
                )
                
                if reached:
                    self.stop_motion()
                    self.get_logger().info(
                        f'✓ Reached {name} in {elapsed:.1f}s! '
                        f'Pos error: {pos_err:.4f}m, Ori error: {ori_err:.4f}rad'
                    )
                    self.get_logger().info(f'Stopping for {self.stop_duration}s...')
                    time.sleep(self.stop_duration)
                    return True
                
                # Track minimum error to see if we're making overall progress
                if pos_err < min_error_seen:
                    min_error_seen = pos_err
                    stuck_count = 0  # Reset if we're improving
                
                # More lenient stuck detection - only trigger if truly stuck
                # Changed from 0.001m to 0.01m threshold and longer time window
                if abs(pos_err - last_pos_error) < 0.01:  # 1cm threshold instead of 1mm
                    stuck_count += 1
                    # Stuck for 5 seconds instead of 2 seconds
                    if stuck_count > 250 and elapsed > 5.0:  # 250 iterations * 0.02s = 5s
                        # Only declare stuck if we're not close to target
                        if pos_err > 0.2:  # More than 20cm away
                            self.get_logger().warn(
                                f'Robot appears stuck at {pos_err:.4f}m error after {elapsed:.1f}s '
                                f'- Moving to next waypoint'
                            )
                            break
                        else:
                            # Close enough - keep trying
                            stuck_count = 0
                else:
                    stuck_count = 0
                
                last_pos_error = pos_err
                
                # Check workspace limits - HARD STOP
                current_distance = np.linalg.norm(current_pos)
                if current_distance > self.max_workspace_radius:
                    self.get_logger().error(
                        f'✗ WORKSPACE LIMIT EXCEEDED! Distance: {current_distance:.3f}m > {self.max_workspace_radius}m'
                    )
                    self.stop_motion()
                    break
                
                # Compute and send servo command
                twist = self.compute_servo_command(current_pos, current_ori, target_pos, target_ori)
                self.twist_pub.publish(twist)
                
                # Log progress every 3 seconds
                if loop_start - last_log_time >= 3.0:
                    remaining = timeout - elapsed
                    self.get_logger().info(
                        f'{name}: Pos error: {pos_err:.4f}m, Ori error: {ori_err:.4f}rad, '
                        f'Distance: {current_distance:.3f}m, Time left: {remaining:.1f}s, '
                        f'Min error: {min_error_seen:.4f}m'
                    )
                    last_log_time = loop_start
                
                # Sleep to maintain loop rate
                loop_elapsed = time.time() - loop_start
                sleep_time = max(0.0, loop_dt - loop_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Timeout or error reached
            self.stop_motion()
            time.sleep(0.5)
        
        # Timeout reached - not considered a failure, just move on
        self.stop_motion()
        self.get_logger().info(f'→ Moving to next waypoint after {name}')
        return False

    def execute_trajectory(self):
        """Execute the full trajectory P1 → P2 → P3"""
        self.get_logger().info('='*60)
        self.get_logger().info('Starting waypoint trajectory: P1 → P2 → P3')
        self.get_logger().info(f'Timeout per waypoint: {self.waypoint_timeout:.0f}s')
        self.get_logger().info('='*60)
        
        total_start = time.time()
        success_count = 0
        
        for i, waypoint in enumerate(self.waypoints):
            self.get_logger().info(f'\n--- Waypoint {i+1}/{len(self.waypoints)} ---')
            success = self.move_to_waypoint(waypoint)
            if success:
                success_count += 1
        
        total_elapsed = time.time() - total_start
        
        self.get_logger().info('='*60)
        self.get_logger().info(f'Trajectory execution complete!')
        self.get_logger().info(
            f'Success: {success_count}/{len(self.waypoints)} waypoints reached'
        )
        self.get_logger().info(f'Total time: {total_elapsed:.1f}s')
        self.get_logger().info('='*60)
        self.stop_motion()
        
        # Return True if at least one waypoint was reached
        return success_count > 0


def main(args=None):
    rclpy.init(args=args)
    
    controller = UR5WaypointServo()
    
    # Wait for TF frames to be available
    if not controller.wait_for_tf_frames(timeout=30.0):
        controller.get_logger().error('Failed to initialize: TF frames not available')
        controller.destroy_node()
        rclpy.shutdown()
        return
    
    # Give system time to stabilize
    time.sleep(2.0)
    
    # Execute trajectory
    success = controller.execute_trajectory()
    
    if success:
        controller.get_logger().info('✓ Mission complete - At least one waypoint reached!')
    else:
        controller.get_logger().warn('⚠️  Mission complete - No waypoints reached (but trajectory executed)')
    
    # Keep node alive for a bit before shutdown
    time.sleep(2.0)
    
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
