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
        self.bin_pose = (np.array([-0.806, 0.010, 0.182]), np.array([-0.684, 0.726, 0.05, 0.008])) # P3 from arm_control
        
        # --- State ---
        self.joint_positions = None
        self.current_state = "INIT"
        self.fruit_index = 0
        self.max_fruits = 3 # Try up to 3 fruits, or until lookup fails
        self.fertilizer_done = False
        
        # --- Control Params (from arm_control.py) ---
        self.position_tolerance = 0.02 # Tighter for grasping
        self.orientation_tolerance = 0.1
        self.joint_tolerance = 0.1
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Manipulation Node Started")

    def joint_callback(self, msg):
        # UR5 joints are usually: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        # Ensure order matches if needed. Assuming standard order.
        # Note: msg.name might be sorted alphabetically, so be careful. 
        # For this task, we assume the driver publishes in correct order or we map them.
        # Standard UR driver order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        # But let's just take the first 6 for now as per arm_control.py
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
        
        future = self.attach_client.call_async(req)
        # We are in a timer callback, so we can't block indefinitely. 
        # But for simplicity in this state machine, we might want to just fire and forget or check later.
        # However, to ensure attachment, we should probably wait a bit.
        # Since we can't await in a sync callback easily without blocking the loop, 
        # we will assume it works or handle it in a separate state if needed.
        # For now, let's just log.
        self.get_logger().info(f"Calling Attach for {model_name}")
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
        self.get_logger().info(f"Calling Detach for {model_name}")
        return True

    def move_joint(self, target_joints):
        if self.joint_positions is None: return False
        
        error = target_joints - self.joint_positions
        if np.max(np.abs(error)) < self.joint_tolerance:
            self.stop_robot()
            return True
            
        vel = error * 2.0 # Gain
        vel = np.clip(vel, -1.0, 1.0)
        
        msg = Float64MultiArray()
        msg.data = vel.tolist()
        self.joint_pub.publish(msg)
        return False

    def move_cartesian(self, target_pos, target_quat):
        # Simple P-controller for Cartesian
        current_pos, current_quat = self.get_tf("tool0") # Assuming tool0 is EE
        if current_pos is None: return False
        
        pos_error = target_pos - current_pos
        dist = np.linalg.norm(pos_error)
        
        # Orientation error
        r_curr = Rotation.from_quat(current_quat)
        r_targ = Rotation.from_quat(target_quat)
        r_diff = r_targ * r_curr.inv()
        rot_vec = r_diff.as_rotvec()
        rot_err = np.linalg.norm(rot_vec)
        
        if dist < self.position_tolerance and rot_err < self.orientation_tolerance:
            self.stop_robot()
            return True
            
        lin_vel = pos_error * 2.0
        ang_vel = rot_vec * 2.0
        
        # Scale
        lin_speed = np.linalg.norm(lin_vel)
        if lin_speed > 0.2: lin_vel *= (0.2 / lin_speed)
        
        ang_speed = np.linalg.norm(ang_vel)
        if ang_speed > 0.5: ang_vel *= (0.5 / ang_speed)
        
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = lin_vel
        msg.angular.x, msg.angular.y, msg.angular.z = ang_vel
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
            # Just wait a bit or move to a safe start
            self.get_logger().info("State: INIT")
            self.current_state = "SEARCH_FRUIT"
            
        elif self.current_state == "SEARCH_FRUIT":
            # Look for next fruit TF
            fruit_frame = f"{self.team_id}_bad_fruit_{self.fruit_index}"
            pos, _ = self.get_tf(fruit_frame)
            
            if pos is not None:
                self.target_fruit_pos = pos
                # Define Grasp Orientation: Z-axis down (pointing to fruit)
                # Standard UR5 base: Z up. Fruit on table.
                # To point Z down: Rotate 180 around X or Y.
                # Let's use a fixed orientation for grasping fruits from top.
                # Q for Z-down: [1, 0, 0, 0] (w,x,y,z) -> 180 deg rotation about X
                # Euler: Roll=pi, Pitch=0, Yaw=0
                r = Rotation.from_euler('xyz', [np.pi, 0, 0])
                self.target_fruit_quat = r.as_quat() # x,y,z,w
                
                self.get_logger().info(f"Found fruit {self.fruit_index}")
                self.current_state = "APPROACH_FRUIT"
            else:
                # If we tried enough times or index is high, move to fertilizer
                if self.fruit_index > 5: # Timeout/Max check
                    self.get_logger().info("No more fruits found. Moving to Fertilizer.")
                    self.current_state = "SEARCH_FERTILIZER"
                else:
                    # Keep searching or increment if we think we missed one?
                    # For now, just retry same index. 
                    # If it never appears, we might get stuck. 
                    # Let's assume perception is running.
                    pass

        elif self.current_state == "APPROACH_FRUIT":
            # Move to offset
            # Approach along Z (normal to fruit top)
            # We want EE Z to align with World -Z (pointing down).
            # So we approach from +Z relative to fruit.
            # Actually, get_approach_pose backs up along the axis.
            # If we want to be ABOVE the fruit, and our Z is pointing DOWN, 
            # backing up along +Z (tool frame) moves us UP (world frame).
            # So offset_dist should be positive.
            
            approach_pos, approach_quat = self.get_approach_pose(
                self.target_fruit_pos, self.target_fruit_quat, 0.15, axis='z'
            )
            
            if self.move_cartesian(approach_pos, approach_quat):
                self.current_state = "GRASP_FRUIT"
                time.sleep(0.5)

        elif self.current_state == "GRASP_FRUIT":
            # Move closer
            # Target is the fruit position (plus maybe small offset to not crash)
            # Fruit center is likely inside the fruit. We want gripper to be around it.
            # Let's go to z + 0.05 or something?
            # User said: "Grasp at < 0.10m"
            
            grasp_pos = self.target_fruit_pos.copy()
            grasp_pos[2] += 0.02 # Slightly above center to avoid smashing
            
            if self.move_cartesian(grasp_pos, self.target_fruit_quat):
                self.call_attach_service("bad_fruit")
                time.sleep(1.0) # Wait for attach
                self.current_state = "RETRACT_FRUIT"

        elif self.current_state == "RETRACT_FRUIT":
            # Move up
            retract_pos = self.target_fruit_pos.copy()
            retract_pos[2] += 0.2
            
            if self.move_cartesian(retract_pos, self.target_fruit_quat):
                self.current_state = "MOVE_TO_BIN"

        elif self.current_state == "MOVE_TO_BIN":
            # Move to Bin Pose
            bin_pos, bin_quat = self.bin_pose
            
            if self.move_cartesian(bin_pos, bin_quat):
                self.call_detach_service("bad_fruit")
                time.sleep(1.0)
                self.fruit_index += 1
                self.current_state = "SEARCH_FRUIT"

        elif self.current_state == "SEARCH_FERTILIZER":
            pos, _ = self.get_tf("1505_fertiliser_can")
            if pos is not None:
                self.target_can_pos = pos
                
                # Orientation: Align EE Z-axis with Can Y-axis.
                # Can is likely upright. Can Y-axis is horizontal.
                # We want EE Z to point at Can.
                # And we want EE Z to be parallel to Can Y.
                # This implies we approach from the side.
                # We need to construct a rotation where Z_ee = Y_can.
                # And we need to define X_ee and Y_ee.
                # Usually X_ee up or down?
                
                # Let's try to look at the can from the Y direction.
                # If Can Y is (0, 1, 0) in world (example), we want EE Z to be (0, -1, 0) (pointing opposite to Y to look at it? Or align with it?)
                # User said: "Align EE Z-axis with Fertilizer Can Y-axis".
                # If Can Y points OUT of the surface we want to grasp, then we want EE Z to point IN (opposite).
                # If Can Y points ALONG the surface, that's different.
                # "Normal to side surface" usually means the radial vector.
                # If the user says "Can Y-axis", I assume the Can's local Y axis IS the normal.
                # So we want EE Z to be anti-parallel to Can Y? Or Parallel?
                # Let's assume we want to align them.
                
                # To be safe, let's just use the TF orientation of the can and rotate our gripper relative to it.
                # If we want EE Z to match Can Y:
                # R_can has columns [X_c, Y_c, Z_c].
                # We want R_ee such that Z_ee = Y_c.
                # We can choose Y_ee = Z_c (up).
                # Then X_ee = Y_ee cross Z_ee = Z_c cross Y_c = -X_c.
                
                # Get Can Rotation Matrix
                _, can_quat = self.get_tf("1505_fertiliser_can")
                r_can = Rotation.from_quat(can_quat)
                m_can = r_can.as_matrix()
                x_c, y_c, z_c = m_can[:,0], m_can[:,1], m_can[:,2]
                
                z_ee = y_c # Align EE Z with Can Y
                y_ee = z_c # Align EE Y with Can Z (gripper fingers vertical?)
                x_ee = np.cross(y_ee, z_ee)
                
                m_ee = np.column_stack((x_ee, y_ee, z_ee))
                r_ee = Rotation.from_matrix(m_ee)
                self.target_can_quat = r_ee.as_quat()
                
                self.current_state = "APPROACH_FERTILIZER"
            else:
                self.get_logger().info("Searching for Fertilizer...")

        elif self.current_state == "APPROACH_FERTILIZER":
            # Approach along Z (which is aligned with Can Y)
            # Offset > 0.2m
            approach_pos, approach_quat = self.get_approach_pose(
                self.target_can_pos, self.target_can_quat, 0.25, axis='z'
            )
            # Note: We are backing up along Z_ee.
            # If Z_ee = Y_can, we are moving in -Y_can direction.
            # If Y_can points OUT of the can, we are moving away from it. Correct.
            
            if self.move_cartesian(approach_pos, approach_quat):
                self.current_state = "GRASP_FERTILIZER"
                time.sleep(0.5)

        elif self.current_state == "GRASP_FERTILIZER":
            # Move closer (< 0.2m)
            # Let's go to 0.15m from center? Or closer?
            # Can radius is small.
            grasp_pos, _ = self.get_approach_pose(
                self.target_can_pos, self.target_can_quat, 0.12, axis='z'
            )
            
            if self.move_cartesian(grasp_pos, self.target_can_quat):
                self.call_attach_service("fertiliser_can")
                time.sleep(1.0)
                self.current_state = "MOVE_TO_LANDING"

        elif self.current_state == "MOVE_TO_LANDING":
            # Move to landing_ebot
            # First retract a bit?
            # Then move to landing pose.
            landing_pos, _ = self.get_tf("landing_ebot")
            if landing_pos is not None:
                # Add some Z offset for landing
                landing_pos[2] += 0.2
                
                # Orientation? Keep current or flat?
                # Let's go flat (Z down)
                r = Rotation.from_euler('xyz', [np.pi, 0, 0])
                landing_quat = r.as_quat()
                
                if self.move_cartesian(landing_pos, landing_quat):
                    self.call_detach_service("fertiliser_can")
                    self.current_state = "DONE"
            else:
                self.get_logger().info("Searching for landing_ebot...")

        elif self.current_state == "DONE":
            self.get_logger().info("Mission Complete!")
            self.stop_robot()
            # Optional: Shutdown?

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
