#!/usr/bin/env python3
"""
Task 2A Shape Detector
Detects geometric shapes (Triangle, Square, Pentagon) using LiDAR data.
Method: Split-and-Merge / RANSAC for line extraction + Geometric analysis.
NO MACHINE LEARNING ALLOWED.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import math
import numpy as np
# NO SKLEARN/ML - Using manual RANSAC implementation

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

class ShapeDetector(Node):

    def __init__(self):
        super().__init__('shape_detector_task2a')
        
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        # Publishers
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.nav_control_pub = self.create_publisher(String, '/nav_control', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        
        # TF Listener with larger cache
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.min_cluster_points = 10
        self.cluster_tolerance = 0.5 
        self.detection_range = 1.5 
        
        # State
        self.is_paused = False
        self.last_detection_time = 0.0
        
        self.get_logger().info('Shape Detector Initialized')

    def transform_point(self, point, source_frame, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            
            p = PointStamped()
            p.header.frame_id = source_frame
            p.point.x = float(point[0])
            p.point.y = float(point[1])
            p.point.z = 0.0
            
            p_transformed = do_transform_point(p, transform)
            return (p_transformed.point.x, p_transformed.point.y)
        except Exception as e:
            self.get_logger().warn(f'TF Error: {e}', throttle_duration_sec=1.0)
            return point # Fallback to relative

    def scan_callback(self, msg):
        if self.is_paused:
            return

        # 1. Preprocess: Filter by range and group into clusters
        clusters = self.cluster_scan(msg)

        # 2. Process each cluster
        for cluster in clusters:
            shape_type, centroid = self.identify_shape(cluster)
            
            if shape_type:
                # Transform centroid to odom
                global_centroid = self.transform_point(centroid, msg.header.frame_id, 'odom')
                self.handle_detection(shape_type, global_centroid)
                break 

    def cluster_scan(self, scan_msg):
        """
        Group points into clusters based on Euclidean distance.
        Returns list of clusters, where each cluster is a list of (x, y) points.
        """
        clusters = []
        current_cluster = []
        
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
        # Filter out inf/nan and out of range
        valid_indices = np.where((ranges > scan_msg.range_min) & (ranges < self.detection_range))[0]
        
        if len(valid_indices) == 0:
            return []

        # Convert to Cartesian
        points = []
        for i in valid_indices:
            r = ranges[i]
            a = angles[i]
            x = r * math.cos(a)
            y = r * math.sin(a)
            points.append((x, y))

        # Simple clustering
        if not points:
            return []
            
        current_cluster.append(points[0])
        
        for i in range(1, len(points)):
            prev_p = points[i-1]
            curr_p = points[i]
            
            dist = math.hypot(curr_p[0] - prev_p[0], curr_p[1] - prev_p[1])
            
            if dist < self.cluster_tolerance:
                current_cluster.append(curr_p)
            else:
                if len(current_cluster) >= self.min_cluster_points:
                    clusters.append(current_cluster)
                current_cluster = [curr_p]
        
        if len(current_cluster) >= self.min_cluster_points:
            clusters.append(current_cluster)
            
        return clusters


    def identify_shape(self, points):
        """
        Identify shape from a cluster of points using classical geometry (NO ML).
        Returns (shape_type, centroid) or (None, None).
        """
        points_np = np.array(points)
        if len(points_np) < 15: 
            return None, None
            
        # Extract line segments using manual RANSAC (no sklearn)
        lines = self.extract_lines_ransac(points_np)
        
        if len(lines) < 2:
            return None, None
        
        centroid = np.mean(points_np, axis=0)
        
        # Analyze geometry to classify shape
        detected_type = self.classify_by_geometry(lines)
        
        if detected_type:
            return detected_type, centroid

        return None, None

    def extract_lines_ransac(self, points, max_lines=4, max_iterations=50, threshold=0.03, min_inliers=10):
        """
        Manual RANSAC implementation for line extraction (NO sklearn).
        Returns list of line parameters.
        """
        lines = []
        remaining = points.copy()
        
        for _ in range(max_lines):
            if len(remaining) < min_inliers:
                break
            
            best_line = None
            best_inliers = []
            
            # RANSAC iterations
            for _ in range(max_iterations):
                if len(remaining) < 2:
                    break
                    
                # Randomly sample 2 points
                idx = np.random.choice(len(remaining), 2, replace=False)
                p1, p2 = remaining[idx]
                
                # Compute line parameters: ax + by + c = 0
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    continue
                
                # Normal form: ax + by + c = 0
                a = -dy
                b = dx
                c = -(a * p1[0] + b * p1[1])
                
                # Normalize
                norm = np.sqrt(a*a + b*b)
                if norm < 1e-6:
                    continue
                a, b, c = a/norm, b/norm, c/norm
                
                # Find inliers
                distances = np.abs(a * remaining[:, 0] + b * remaining[:, 1] + c)
                inlier_mask = distances < threshold
                inliers = remaining[inlier_mask]
                
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_line = {'a': a, 'b': b, 'c': c}
            
            if best_line and len(best_inliers) >= min_inliers:
                # Compute line direction vector
                v = np.array([best_line['b'], -best_line['a']])  # perpendicular to normal
                v = v / np.linalg.norm(v)
                best_line['v'] = v
                lines.append(best_line)
                
                # Remove inliers
                distances = np.abs(best_line['a'] * remaining[:, 0] + 
                                 best_line['b'] * remaining[:, 1] + best_line['c'])
                remaining = remaining[distances >= threshold]
            else:
                break
        
        return lines

    def classify_by_geometry(self, lines):
        """
        Classify shape based on detected line segments and angles.
        Triangle: 60° angles
        Square: 90° angles  
        Pentagon: 108° angles
        """
        num_lines = len(lines)
        
        if num_lines < 2:
            return None
            
        # Calculate angle between first two prominent lines
        v1 = lines[0]['v']
        v2 = lines[1]['v']
        
        # Dot product: cos(θ) = v1 · v2
        dot_prod = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(np.abs(dot_prod))  # Use abs to get acute angle
        angle_deg = np.degrees(angle_rad)
        
        # Check both angle and its supplement
        possible_angles = [angle_deg, 180 - angle_deg]
        
        for a in possible_angles:
            if 85 <= a <= 95:  # ~90° → Square
                return 'BAD_HEALTH'
            elif 55 <= a <= 65:  # ~60° → Triangle
                return 'FERTILIZER_REQUIRED'
            elif 103 <= a <= 113:  # ~108° → Pentagon
                return 'DOCK_STATION'
        
        # If 3+ lines visible, likely Pentagon
        if num_lines >= 3:
            return 'DOCK_STATION'
        
        return None

    def handle_detection(self, shape_type, centroid):
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_detection_time < 5.0: # Debounce
            return

        self.get_logger().info(f'Detected {shape_type} at ({centroid[0]:.2f}, {centroid[1]:.2f})')
        
        # Publish control PAUSE
        self.nav_control_pub.publish(String(data='PAUSE'))
        self.is_paused = True
        self.last_detection_time = current_time
        
        # Publish status
        # Format: Status,x,y
        # Note: Centroid is in LiDAR frame (relative to robot).
        # Task asks for detection position. Usually global or relative?
        # "Examples: FERTILIZER_REQUIRED,-1.2,2.5" -> These look like global coordinates.
        # We need to transform centroid to global frame?
        # The task description says: "Status,x,y: Detection position coordinates".
        # If we need global, we need TF.
        # Let's assume global for now and add TF lookup if needed. 
        # Wait, I don't have TF listener here yet.
        # Let's add TF listener to this node too.
        
        # For now, publishing relative might be wrong.
        # I will add TF listener in next step if needed. 
        # But let's just publish what we have and maybe fix later.
        # Actually, let's assume the user wants the location of the shape in the map.
        
        msg = f'{shape_type},{centroid[0]:.2f},{centroid[1]:.2f}'
        self.detection_pub.publish(String(data=msg))
        
        # Timer to resume
        self.resume_timer = self.create_timer(2.0, self.resume_navigation)
        
    def resume_navigation(self):
        self.get_logger().info('Resuming navigation...')
        self.nav_control_pub.publish(String(data='RESUME'))
        self.is_paused = False
        self.resume_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
