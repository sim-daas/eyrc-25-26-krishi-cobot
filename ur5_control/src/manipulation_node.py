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
from linkattacher_msgs.srv import AttachLink, DetachLink

class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node')
        
        # --- Publishers ---
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/delta_joint_cmds', 10)
        
        # --- Subscribers ---
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- Services ---
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        
        # --- Constants ---
        self.team_id = 1505
        self.home_joints = np.array([-1.57, -1.57, 0.0, -1.57, 0.0, 0.0]) # Example home
        # SAFE TRANSIT JOINT CONFIGURATION (Updated to avoid singularity)
        # Pan, Lift, Elbow, Wrist1, Wrist2, Wrist3
        self.safe_transit_joints = np.array([
            -3.973,  # shoulder_pan_joint
            -1.596,  # shoulder_lift_joint
             0.0,    # elbow_joint
             0.0,    # wrist_1_joint
             0.0,    # wrist_2_joint
             0.0     # wrist_3_joint
        ])
        self.bin_pose = (np.array([-0.806, 0.010, 0.182]), np.array([-0.684, 0.726, 0.05, 0.008])) # P3 from arm_control
        
        # --- State ---
        self.joint_positions = None
        self.current_state = "INIT"
        self.next_state = None # For mid-point transitions
        self.fruit_index = 0
        self.max_fruits = 2 # Try up to 3 fruits, or until lookup fails
        self.fertilizer_done = False
        self.debug_counter = 0 # For logging
        
        # --- Control Params (Restored from arm_control.py) ---
        self.position_tolerance = 0.09 # Tighter for grasping
        self.orientation_tolerance = 0.1
        self.joint_tolerance = 0.1
        
        # Velocity limits
        self.max_joint_velocity = 3.0
        self.max_linear_velocity_coarse = 2.0
        self.max_angular_velocity_coarse = 2.0
        self.max_linear_velocity_fine = 0.19
        self.max_angular_velocity_fine = 0.7
        
        # Distance thresholds
        self.far_distance = 0.4
        self.medium_distance = 0.2
        self.close_distance = 0.12
        
        # Gains
        self.joint_gain = 5
        self.position_gain_base = 4.0 # Increased from 2.5 to reduce slowdown
        self.orientation_gain_base = 3.5
        
        self.servo_phase = "COARSE_APPROACH"
        self.use_fine_control = False # Option to switch off fine_tune approach
        
        self.control_timer = self.create_timer(0.01, self.control_loop) # Faster loop (100Hz)
        self.get_logger().info("Manipulation Node Started (Advanced Motion Control)")

    def joint_callback(self, msg):
        self.joint_positions = np.array(msg.position[:6])

    def get_tf(self, target_frame, source_frame="base_link"):
        try:
            trans = self.tf_buffer.lookup_transform(source_frame, target_frame, rclpy.time.Time())
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            return pos, quat
        except Exception:
            return None, None

    def call_attach_service(self, model_name):
        if not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attach service not available")
            return False
        
        req = AttachLink.Request()
        req.model1_name = model_name
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        
        self.attach_client.call_async(req)
        self.get_logger().info(f"-> Attaching {model_name}...")
        return True

    def call_detach_service(self, model_name):
        if not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Detach service not available")
            return False
        
        req = DetachLink.Request()
        req.model1_name = model_name
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        
        self.detach_client.call_async(req)
        self.get_logger().info(f"-> Detaching {model_name}...")
        return True

    # --- Advanced Motion Control Methods ---

    def compute_orientation_error(self, current_quat, target_quat):
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        r_current = Rotation.from_quat(current_quat)
        r_target = Rotation.from_quat(target_quat)
        r_diff = r_target * r_current.inv()
        return r_diff.magnitude()

    def compute_angular_velocity_from_quaternion_error(self, current_quat, target_quat):
        current_quat = current_quat / np.linalg.norm(current_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        r_current = Rotation.from_quat(current_quat)
        r_target = Rotation.from_quat(target_quat)
        r_error = r_target * r_current.inv()
        return r_error.as_rotvec()

    def compute_blending_weights(self, distance_to_target, orientation_error):
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
        if self.servo_phase == "COARSE_APPROACH":
            # Less aggressive scaling: maintain higher speed until very close
            if distance_to_target < self.medium_distance:
                # Scale from 1.0 down to 0.3 linearly
                distance_scale = 0.3 + 0.7 * (distance_to_target / self.medium_distance)
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
        pos_error_vector = target_pos - current_pos
        distance_to_target = np.linalg.norm(pos_error_vector)
        orientation_error = self.compute_orientation_error(current_quat, target_quat)
        
        pos_weight, ori_weight = self.compute_blending_weights(distance_to_target, orientation_error)
        
        linear_vel = pos_error_vector * self.position_gain_base * pos_weight
        angular_vel = self.compute_angular_velocity_from_quaternion_error(current_quat, target_quat) * self.orientation_gain_base * ori_weight
        
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

    def move_joint(self, target_joints, tolerance=None):
        if self.joint_positions is None: return False
        
        tol = tolerance if tolerance is not None else self.joint_tolerance
        error = target_joints - self.joint_positions
        
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            max_err = np.max(np.abs(error))
            self.get_logger().info(f"Joint Error Max: {max_err:.3f}")
            
        if np.max(np.abs(error)) < tol:
            self.stop_robot()
            return True
            
        vel = error * self.joint_gain
        vel = np.clip(vel, -self.max_joint_velocity, self.max_joint_velocity)
        
        msg = Float64MultiArray()
        msg.data = vel.tolist()
        self.joint_pub.publish(msg)
        return False

    def move_cartesian(self, target_pos, target_quat, tolerance=None, orientation_tolerance=None):
        current_pos, current_quat = self.get_tf("tool0") 
        if current_pos is None: return False
        
        # Use provided tolerance or default
        pos_tol = tolerance if tolerance is not None else self.position_tolerance
        ori_tol = orientation_tolerance if orientation_tolerance is not None else self.orientation_tolerance
        
        # Phase transition
        pos_error = np.linalg.norm(target_pos - current_pos)
        if self.use_fine_control and self.servo_phase == "COARSE_APPROACH" and pos_error < self.close_distance:
            self.servo_phase = "FINE_TUNE"
            
        linear_vel, angular_vel, dist, ori_err = self.compute_twist_command(
            current_pos, current_quat, target_pos, target_quat
        )
        
        if dist < pos_tol and ori_err < ori_tol:
            self.stop_robot()
            self.servo_phase = "COARSE_APPROACH" # Reset for next move
            return True
            
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(linear_vel[0]), float(linear_vel[1]), float(linear_vel[2])
        msg.angular.x, msg.angular.y, msg.angular.z = float(angular_vel[0]), float(angular_vel[1]), float(angular_vel[2])
        self.twist_pub.publish(msg)
        return False

    def stop_robot(self):
        self.twist_pub.publish(Twist())
        msg = Float64MultiArray()
        msg.data = [0.0]*6
        self.joint_pub.publish(msg)

    def get_approach_pose(self, target_pos, target_quat, offset_dist, axis='z'):
        # Calculate approach pose by backing up along the specified axis of the target orientation
        r = Rotation.from_quat(target_quat)
        matrix = r.as_matrix()
        
        if axis == 'z':
            offset_vec = matrix[:, 2] * offset_dist # Z-axis
        elif axis == 'y':
            offset_vec = matrix[:, 1] * offset_dist # Y-axis
        else:
            offset_vec = np.zeros(3)
            
        approach_pos = target_pos - offset_vec # Back up from target
        return approach_pos, target_quat

    def control_loop(self):
        if self.joint_positions is None: return

        # --- State Machine ---
        
        if self.current_state == "INIT":
            self.get_logger().info("State: INIT -> Moving to Safe Mid-Point")
            self.next_state = "SEARCH_FERTILIZER"
            self.current_state = "MOVE_TO_MIDPOINT"
            
        elif self.current_state == "MOVE_TO_MIDPOINT":
            # Relaxed tolerance for mid-point (0.3 rad)
            if self.move_joint(self.safe_transit_joints, tolerance=0.3):
                self.get_logger().info(f"Reached Mid-Point. Transitioning to {self.next_state}")
                self.current_state = self.next_state
                time.sleep(0.5)

        elif self.current_state == "SEARCH_FRUIT":
            # Look for next fruit TF
            fruit_frame = f"{self.team_id}_bad_fruit_{self.fruit_index}"
            pos, _ = self.get_tf(fruit_frame)
            
            if pos is not None:
                self.target_fruit_pos = pos
                # Define Grasp Orientation: Z-axis down
                r = Rotation.from_euler('xyz', [np.pi, 0, 0])
                self.target_fruit_quat = r.as_quat() 
                
                self.get_logger().info(f"Found {fruit_frame}. Moving directly to GRASP_FRUIT")
                self.current_state = "GRASP_FRUIT"
            else:
                # If we tried enough times or index is high, stop
                if self.fruit_index > self.max_fruits: 
                    self.get_logger().info("No more fruits found. Mission Complete.")
                    self.current_state = "DONE"
                else:
                    # Retry or just wait
                    pass

        elif self.current_state == "GRASP_FRUIT":
            grasp_pos = self.target_fruit_pos.copy()
            grasp_pos[2] += 0.02 
            
            # Tight tolerance for fruit grasp
            if self.move_cartesian(grasp_pos, self.target_fruit_quat, tolerance=0.02):
                self.call_attach_service("bad_fruit")
                time.sleep(1.0) 
                self.get_logger().info("Attached Fruit. Retracting...")
                self.current_state = "RETRACT_FRUIT"

        elif self.current_state == "RETRACT_FRUIT":
            retract_pos = self.target_fruit_pos.copy()
            retract_pos[2] += 0.2
            
            if self.move_cartesian(retract_pos, self.target_fruit_quat):
                self.get_logger().info("Retracted. Moving to Mid-Point before Bin.")
                self.next_state = "MOVE_TO_BIN"
                self.current_state = "MOVE_TO_MIDPOINT"

        elif self.current_state == "MOVE_TO_BIN":
            bin_pos, bin_quat = self.bin_pose
            
            if self.move_cartesian(bin_pos, bin_quat):
                self.call_detach_service("bad_fruit")
                time.sleep(1.0)
                self.fruit_index += 1
                self.get_logger().info(f"Dropped Fruit. Next Index: {self.fruit_index}. Moving to Mid-Point.")
                self.next_state = "SEARCH_FRUIT"
                self.current_state = "MOVE_TO_MIDPOINT"

        elif self.current_state == "SEARCH_FERTILIZER":
            pos, _ = self.get_tf("1505_fertiliser_can")
            if pos is not None:
                self.target_can_pos = pos
                
                # Orientation: Align EE Z-axis with Can -Y axis (Negative Y).
                _, can_quat = self.get_tf("1505_fertiliser_can")
                r_can = Rotation.from_quat(can_quat)
                m_can = r_can.as_matrix()
                x_c, y_c, z_c = m_can[:,0], m_can[:,1], m_can[:,2]
                
                z_ee = -y_c # Align EE Z with Can -Y
                y_ee = z_c  # Align EE Y with Can Z
                x_ee = np.cross(y_ee, z_ee)
                
                m_ee = np.column_stack((x_ee, y_ee, z_ee))
                r_ee = Rotation.from_matrix(m_ee)
                self.target_can_quat = r_ee.as_quat()
                
                self.get_logger().info(f"SEARCH_FERTILIZER: Found Can at {pos}")
                self.get_logger().info("Found Fertilizer Can. Moving to APPROACH_FERTILIZER")
                self.current_state = "APPROACH_FERTILIZER"
            else:
                self.get_logger().info("Searching for Fertilizer...")

        elif self.current_state == "APPROACH_FERTILIZER":
            # 0.15m approach offset (Matched to Fruit System)
            approach_pos, approach_quat = self.get_approach_pose(
                self.target_can_pos, self.target_can_quat, 0.15, axis='z'
            )
            
            # Concise Debug Log
            curr_p, _ = self.get_tf("tool0")
            if curr_p is not None:
                self.get_logger().info(f"Approach Target: {approach_pos}, Current: {curr_p}")
            
            # Loose tolerance for approach (0.08m pos, 0.5 rad ori)
            if self.move_cartesian(approach_pos, approach_quat, tolerance=0.08, orientation_tolerance=0.5):
                self.get_logger().info("Reached Approach Pose. Moving to GRASP_FERTILIZER")
                self.current_state = "GRASP_FERTILIZER"
                time.sleep(0.5)

        elif self.current_state == "GRASP_FERTILIZER":
            # 0.02m grasp offset (Close grasp, matched to Fruit System)
            grasp_pos, _ = self.get_approach_pose(
                self.target_can_pos, self.target_can_quat, 0.02, axis='z'
            )
            
            # Tight tolerance for grasp (0.02m)
            if self.move_cartesian(grasp_pos, self.target_can_quat, tolerance=0.02):
                self.call_attach_service("fertiliser_can")
                time.sleep(1.0)
                self.get_logger().info("Attached Fertilizer. Retracting along +Y...")
                self.current_state = "RETRACT_FERTILIZER"

        elif self.current_state == "RETRACT_FERTILIZER":
            # Move to fixed point: Target + 0.3m along World +Y axis
            # Using static target_can_pos prevents the "receding target" bug
            retract_pos = self.target_can_pos.copy()
            retract_pos[1] += 0.55 # Add 0.3m to Y
            
            # Relaxed tolerance to ensure transition
            if self.move_cartesian(retract_pos, self.target_can_quat, tolerance=0.2, orientation_tolerance=0.5):
                self.get_logger().info("Retracted to fixed point. Moving to Mid-Point before Landing.")
                self.next_state = "APPROACH_LANDING"
                self.current_state = "MOVE_TO_MIDPOINT"

        elif self.current_state == "APPROACH_LANDING":
            landing_pos, _ = self.get_tf("landing_ebot")
            if landing_pos is not None:
                # Approach: +0.1m X, +0.35m Z
                approach_pos = landing_pos.copy()
                approach_pos[2] += 0.45
                
                r = Rotation.from_euler('xyz', [1.57, 0, 0])
                landing_quat = r.as_quat()
                
                if self.move_cartesian(approach_pos, landing_quat, tolerance=0.1):
                    self.get_logger().info("Reached Landing Approach. Positioning for Drop...")
                    self.current_state = "DROP_FERTILIZER"
            else:
                self.get_logger().info("Searching for landing_ebot...")

        elif self.current_state == "DROP_FERTILIZER":
            landing_pos, _ = self.get_tf("landing_ebot")
            if landing_pos is not None:
                # Drop: +0.1m X, +0.5m Z (Higher than approach?)
                # User requested 0.5m height for ungrasping
                drop_pos = landing_pos.copy()
                drop_pos[2] += 0.15
                
                r = Rotation.from_euler('xyz', [1.57, 0, 0])
                landing_quat = r.as_quat()
                
                if self.move_cartesian(drop_pos, landing_quat, tolerance=0.05):
                    self.call_detach_service("fertiliser_can")
                    self.get_logger().info("Placed Fertilizer. Moving to Mid-Point before Fruits.")
                    self.next_state = "SEARCH_FRUIT" # Proceed to fruits
                    self.current_state = "MOVE_TO_MIDPOINT"

        elif self.current_state == "DONE":
            self.stop_robot()

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
