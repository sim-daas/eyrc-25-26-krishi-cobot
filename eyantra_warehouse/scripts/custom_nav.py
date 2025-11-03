#!/usr/bin/env python3
"""
Custom Navigation Node for Gazebo Simulation
Implements waypoint-based navigation with corner and target waypoints
Uses PID control for path tracking and state machine for navigation logic
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
import math
from collections import deque
import time


class CustomNavigator(Node):
    """
    Custom navigation node for structured environment navigation.
    Navigates through predefined waypoints using corner and target waypoints.
    """
    
    # ============================================================
    # CONSTANTS - Modify these for tuning
    # ============================================================
    
    # Waypoint Database
    WAYPOINTS_DB = {
        # Corner Waypoints (no yaw requirement)
        'C1': {'type': 'corner', 'pos': (0.0, 0.0), 'yaw': None},
        'C2': {'type': 'corner', 'pos': (0.3, -1.79), 'yaw': None},
        'C3': {'type': 'corner', 'pos': (7.77, -1.79), 'yaw': None},
        'C4': {'type': 'corner', 'pos': (7.77, -0.07), 'yaw': None},
        
        # Target Waypoints (with specific yaw)
        'P1': {'type': 'target', 'pos': (4.66, -1.79), 'yaw': 0.0},
        'P2': {'type': 'target', 'pos': (5.94, -0.05), 'yaw': -3.14},
        'P3': {'type': 'target', 'pos': (0.0, 0.0), 'yaw': -3.14},
    }
    
    # Graph Connectivity (adjacency list)
    GRAPH = {
        'C1': ['C2'],
        'C2': ['C1', 'P1'],
        'P1': ['C2', 'C3'],
        'C3': ['P1', 'C4'],
        'C4': ['C3', 'P2'],
        'P2': ['C4', 'P3'],
        'P3': ['P2'],
    }
    
    # Navigation Parameters
    LINEAR_VEL = 0.3          # m/s
    ANGULAR_VEL = 0.5         # rad/s
    
    # PID Controller Gains
    KP_CROSSTRACK = 1.5       # Proportional gain for crosstrack error
    KI_CROSSTRACK = 0.0       # Integral gain
    KD_CROSSTRACK = 0.1       # Derivative gain
    
    KP_HEADING = 2.0          # Proportional gain for heading error
    
    # Tolerances
    WAYPOINT_TOLERANCE = 0.2  # meters
    HEADING_TOLERANCE = math.radians(10)  # radians (~10 degrees)
    SMOOTH_TURN_THRESHOLD = math.radians(20)  # radians (~20 degrees)
    
    # Safety
    SAFETY_STOP_DISTANCE = 0.05  # meters (5cm)
    
    # Timing
    TARGET_WAIT_TIME = 0.5    # seconds
    CONTROL_LOOP_RATE = 10.0  # Hz
    
    # Mission Definition (P1 -> P2 -> P3)
    MISSION_SEQUENCE = ['P1', 'P2', 'P3']
    
    def __init__(self):
        super().__init__('custom_navigator')
        
        # ============================================================
        # ROS Setup
        # ============================================================
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        
        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.CONTROL_LOOP_RATE, self.navigation_loop
        )
        
        # ============================================================
        # State Variables
        # ============================================================
        
        # Mission state
        self.mission_index = 0  # Current target in MISSION_SEQUENCE
        self.global_path = []   # List of waypoint names to follow
        self.path_index = 0     # Current waypoint in global_path
        
        # Navigation state machine
        self.state = 'IDLE'  # IDLE, TURNING_TO_NEXT, DRIVING, ARRIVED, 
                             # PERFORMING_CUSTOM_YAW, WAITING
        
        # Current robot pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Safety
        self.obstacle_detected = False
        self.min_obstacle_distance = float('inf')
        
        # PID controller state
        self.prev_crosstrack_error = 0.0
        self.integral_crosstrack_error = 0.0
        self.prev_time = self.get_clock().now()
        
        # Wait timer
        self.wait_start_time = None
        
        # ============================================================
        # Initialization
        # ============================================================
        
        self.get_logger().info('Custom Navigator initialized')
        self.get_logger().info(f'Mission sequence: {self.MISSION_SEQUENCE}')
        
        # Wait for TF to be ready
        time.sleep(1.0)
        
        # Plan the complete mission path
        if not self.plan_mission():
            self.get_logger().error('Failed to plan mission. Shutting down.')
            return
        
        # Start the mission
        self.state = 'TURNING_TO_NEXT'
        self.get_logger().info(f'Mission started. Path: {self.global_path}')
    
    # ============================================================
    # Path Planning
    # ============================================================
    
    def plan_mission(self):
        """
        Plan the complete path for the entire mission sequence.
        Returns True if successful, False otherwise.
        """
        # Get starting position
        if not self.update_robot_pose():
            self.get_logger().error('Cannot get robot pose for planning')
            return False
        
        start_waypoint = self.find_nearest_waypoint(self.current_x, self.current_y)
        if not start_waypoint:
            self.get_logger().error('Cannot find starting waypoint')
            return False
        
        self.get_logger().info(f'Starting at waypoint: {start_waypoint}')
        
        # Plan path through all mission targets
        complete_path = []
        current_start = start_waypoint
        
        for target_name in self.MISSION_SEQUENCE:
            segment_path = self.dijkstra_shortest_path(current_start, target_name)
            
            if not segment_path:
                self.get_logger().error(
                    f'No path found from {current_start} to {target_name}'
                )
                return False
            
            # Add segment (excluding the first node if it's not the very first)
            if complete_path:
                segment_path = segment_path[1:]  # Skip duplicate start node
            
            complete_path.extend(segment_path)
            current_start = target_name
        
        self.global_path = complete_path
        self.path_index = 0
        
        self.get_logger().info(f'Complete mission path planned: {complete_path}')
        return True
    
    def find_nearest_waypoint(self, x, y):
        """
        Find the nearest waypoint to the given position.
        Special handling: If at (0,0), prioritize C1 over P3.
        """
        min_dist = float('inf')
        nearest = None
        
        for name, data in self.WAYPOINTS_DB.items():
            wx, wy = data['pos']
            dist = math.hypot(x - wx, y - wy)
            
            if dist < min_dist:
                min_dist = dist
                nearest = name
        
        # Special case: If very close to (0,0), prefer C1 (corner) over P3 (target)
        if min_dist < 0.1:  # Within 10cm of a waypoint
            if nearest in ['C1', 'P3']:
                # We're at (0,0), prefer C1 as starting point
                nearest = 'C1'
                self.get_logger().info('At (0,0): Selected C1 as starting waypoint')
        
        return nearest
    
    def dijkstra_shortest_path(self, start, goal):
        """
        Find shortest path from start to goal using Dijkstra's algorithm.
        Returns list of waypoint names, or None if no path exists.
        """
        if start == goal:
            return [start]
        
        # Initialize
        distances = {node: float('inf') for node in self.GRAPH}
        distances[start] = 0
        previous = {node: None for node in self.GRAPH}
        unvisited = set(self.GRAPH.keys())
        
        while unvisited:
            # Find node with minimum distance
            current = min(unvisited, key=lambda node: distances[node])
            
            if distances[current] == float('inf'):
                break  # No path exists
            
            if current == goal:
                break  # Found goal
            
            unvisited.remove(current)
            
            # Update distances to neighbors
            for neighbor in self.GRAPH[current]:
                if neighbor in unvisited:
                    # Calculate distance (Euclidean distance between waypoints)
                    cx, cy = self.WAYPOINTS_DB[current]['pos']
                    nx, ny = self.WAYPOINTS_DB[neighbor]['pos']
                    edge_distance = math.hypot(nx - cx, ny - cy)
                    
                    alt_distance = distances[current] + edge_distance
                    
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        if distances[goal] == float('inf'):
            return None  # No path found
        
        path = []
        current = goal
        while current is not None:
            path.insert(0, current)
            current = previous[current]
        
        return path
    
    # ============================================================
    # Robot Pose Management
    # ============================================================
    
    def update_robot_pose(self):
        """
        Get the current robot pose from TF.
        Returns True if successful, False otherwise.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'ebot_base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            
            # Extract position
            self.current_x = transform.transform.translation.x
            self.current_y = transform.transform.translation.y
            
            # Extract yaw from quaternion
            quat = transform.transform.rotation
            self.current_yaw = self.quaternion_to_yaw(quat)
            
            return True
            
        except Exception as e:
            self.get_logger().warn(f'Failed to get robot pose: {e}')
            return False
    
    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    # ============================================================
    # Sensor Callbacks
    # ============================================================
    
    def scan_callback(self, msg):
        """Process Lidar scan for obstacle detection."""
        # Check all ranges (360 degrees)
        valid_ranges = [r for r in msg.ranges if not math.isinf(r) and not math.isnan(r)]
        
        if valid_ranges:
            self.min_obstacle_distance = min(valid_ranges)
            self.obstacle_detected = self.min_obstacle_distance < self.SAFETY_STOP_DISTANCE
        else:
            self.min_obstacle_distance = float('inf')
            self.obstacle_detected = False
    
    # ============================================================
    # Main Navigation Loop (State Machine)
    # ============================================================
    
    def navigation_loop(self):
        """Main control loop called at CONTROL_LOOP_RATE Hz."""
        # Update robot pose
        if not self.update_robot_pose():
            return
        
        # Safety check
        if self.obstacle_detected:
            self.get_logger().warn(
                f'EMERGENCY STOP! Obstacle at {self.min_obstacle_distance:.2f}m',
                throttle_duration_sec=1.0
            )
            self.publish_velocity(0.0, 0.0)
            return
        
        # State machine
        if self.state == 'IDLE':
            self.handle_idle_state()
        
        elif self.state == 'TURNING_TO_NEXT':
            self.handle_turning_state()
        
        elif self.state == 'DRIVING':
            self.handle_driving_state()
        
        elif self.state == 'ARRIVED':
            self.handle_arrived_state()
        
        elif self.state == 'PERFORMING_CUSTOM_YAW':
            self.handle_custom_yaw_state()
        
        elif self.state == 'WAITING':
            self.handle_waiting_state()
    
    def handle_idle_state(self):
        """Handle IDLE state - do nothing."""
        self.publish_velocity(0.0, 0.0)
    
    def handle_turning_state(self):
        """Handle TURNING_TO_NEXT state - rotate to face next waypoint."""
        if self.path_index >= len(self.global_path):
            # Mission complete
            self.get_logger().info('Mission complete!')
            self.state = 'IDLE'
            self.publish_velocity(0.0, 0.0)
            return
        
        # Get target waypoint
        target_name = self.global_path[self.path_index]
        target_x, target_y = self.WAYPOINTS_DB[target_name]['pos']
        
        # Calculate desired heading
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        target_yaw = math.atan2(dy, dx)
        
        # Calculate angular error
        angular_error = self.normalize_angle(target_yaw - self.current_yaw)
        
        # Check if aligned
        if abs(angular_error) < self.HEADING_TOLERANCE:
            self.get_logger().info(
                f'Aligned to waypoint {target_name}. Starting drive.'
            )
            self.state = 'DRIVING'
            self.publish_velocity(0.0, 0.0)
            return
        
        # Smooth turn if error is small
        if abs(angular_error) < self.SMOOTH_TURN_THRESHOLD:
            # Proportional turn
            angular_vel = self.KP_HEADING * angular_error
            angular_vel = self.clamp(angular_vel, -self.ANGULAR_VEL, self.ANGULAR_VEL)
        else:
            # Full speed turn
            angular_vel = self.ANGULAR_VEL if angular_error > 0 else -self.ANGULAR_VEL
        
        self.publish_velocity(0.0, angular_vel)
    
    def handle_driving_state(self):
        """Handle DRIVING state - follow path to next waypoint using PID."""
        if self.path_index >= len(self.global_path):
            self.state = 'IDLE'
            return
        
        # Get target waypoint
        target_name = self.global_path[self.path_index]
        target_x, target_y = self.WAYPOINTS_DB[target_name]['pos']
        
        # Calculate distance to target
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.hypot(dx, dy)
        
        # Check if arrived
        if distance < self.WAYPOINT_TOLERANCE:
            self.get_logger().info(f'Arrived at waypoint: {target_name}')
            self.state = 'ARRIVED'
            self.publish_velocity(0.0, 0.0)
            self.reset_pid()
            return
        
        # Path tracking with PID
        # Get previous waypoint for line definition
        if self.path_index > 0:
            prev_name = self.global_path[self.path_index - 1]
            prev_x, prev_y = self.WAYPOINTS_DB[prev_name]['pos']
        else:
            # First segment - use current position as "previous"
            prev_x, prev_y = self.current_x, self.current_y
        
        # Calculate crosstrack error
        crosstrack_error = self.calculate_crosstrack_error(
            prev_x, prev_y, target_x, target_y, self.current_x, self.current_y
        )
        
        # Calculate heading error
        desired_yaw = math.atan2(target_y - prev_y, target_x - prev_x)
        heading_error = self.normalize_angle(desired_yaw - self.current_yaw)
        
        # PID controller for angular velocity
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time
        
        if dt > 0:
            # PID terms
            self.integral_crosstrack_error += crosstrack_error * dt
            derivative_error = (crosstrack_error - self.prev_crosstrack_error) / dt
            
            # Calculate correction
            angular_correction = (
                self.KP_CROSSTRACK * crosstrack_error +
                self.KI_CROSSTRACK * self.integral_crosstrack_error +
                self.KD_CROSSTRACK * derivative_error
            )
            
            # Add heading error correction
            angular_vel = angular_correction + self.KP_HEADING * heading_error
            
            # Clamp angular velocity
            angular_vel = self.clamp(angular_vel, -self.ANGULAR_VEL, self.ANGULAR_VEL)
            
            self.prev_crosstrack_error = crosstrack_error
        else:
            angular_vel = 0.0
        
        # Publish velocity
        self.publish_velocity(self.LINEAR_VEL, angular_vel)
        
        # Debug logging
        self.get_logger().info(
            f'Driving to {target_name}: dist={distance:.2f}m, '
            f'xtalk={crosstrack_error:.3f}, head_err={math.degrees(heading_error):.1f}°',
            throttle_duration_sec=1.0
        )
    
    def handle_arrived_state(self):
        """Handle ARRIVED state - decide next action based on waypoint type."""
        current_waypoint_name = self.global_path[self.path_index]
        current_waypoint = self.WAYPOINTS_DB[current_waypoint_name]
        
        if current_waypoint['type'] == 'target':
            # Target waypoint - perform custom yaw
            self.get_logger().info(
                f'At target {current_waypoint_name}. Performing custom yaw alignment.'
            )
            self.state = 'PERFORMING_CUSTOM_YAW'
        else:
            # Corner waypoint - skip to next
            self.get_logger().info(
                f'At corner {current_waypoint_name}. Moving to next waypoint.'
            )
            self.path_index += 1
            self.state = 'TURNING_TO_NEXT'
    
    def handle_custom_yaw_state(self):
        """Handle PERFORMING_CUSTOM_YAW state - rotate to target's specific yaw."""
        current_waypoint_name = self.global_path[self.path_index]
        target_yaw = self.WAYPOINTS_DB[current_waypoint_name]['yaw']
        
        # Calculate angular error
        angular_error = self.normalize_angle(target_yaw - self.current_yaw)
        
        # Check if aligned
        if abs(angular_error) < self.HEADING_TOLERANCE:
            self.get_logger().info(
                f'Custom yaw achieved at {current_waypoint_name}. Waiting...'
            )
            self.state = 'WAITING'
            self.wait_start_time = self.get_clock().now()
            self.publish_velocity(0.0, 0.0)
            return
        
        # Rotate to target yaw
        angular_vel = self.ANGULAR_VEL if angular_error > 0 else -self.ANGULAR_VEL
        self.publish_velocity(0.0, angular_vel)
        
        self.get_logger().info(
            f'Aligning to custom yaw: error={math.degrees(angular_error):.1f}°',
            throttle_duration_sec=0.5
        )
    
    def handle_waiting_state(self):
        """Handle WAITING state - pause for TARGET_WAIT_TIME seconds."""
        elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
        
        if elapsed >= self.TARGET_WAIT_TIME:
            self.get_logger().info('Wait complete. Moving to next waypoint.')
            self.path_index += 1
            self.state = 'TURNING_TO_NEXT'
        
        self.publish_velocity(0.0, 0.0)
    
    # ============================================================
    # Utility Functions
    # ============================================================
    
    def calculate_crosstrack_error(self, x1, y1, x2, y2, xr, yr):
        """
        Calculate crosstrack error (perpendicular distance from robot to line).
        Positive error means robot is to the left of the path.
        """
        # Vector from point 1 to point 2 (path direction)
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from point 1 to robot
        drx = xr - x1
        dry = yr - y1
        
        # Length of path segment
        path_length = math.hypot(dx, dy)
        
        if path_length < 0.001:  # Avoid division by zero
            return 0.0
        
        # Crosstrack error (signed distance)
        crosstrack = (dx * dry - dy * drx) / path_length
        
        return crosstrack
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def clamp(self, value, min_val, max_val):
        """Clamp value between min and max."""
        return max(min_val, min(value, max_val))
    
    def reset_pid(self):
        """Reset PID controller state."""
        self.prev_crosstrack_error = 0.0
        self.integral_crosstrack_error = 0.0
    
    def publish_velocity(self, linear, angular):
        """Publish velocity command."""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        navigator = CustomNavigator()
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
