#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, PointStamped
from tf_transformations import quaternion_from_euler
import tf2_geometry_msgs

# Runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

class PerceptionNode(Node):
    """
    Perception Node for Krishi coBot Task 1B.
    Handles:
    1. Bad Fruit Detection (HSV + Sorting)
    2. ArUco Marker Detection (Fertilizer Can + Landing Zone)
    3. TF Broadcasting
    """

    def __init__(self):
        super().__init__("perception_node")
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.image_stamp = None
        self.team_id = 1505

        # --- Fruit Detection Params ---
        self.lower_white = np.array([0, 0, 65], dtype=np.int32)
        self.upper_white = np.array([180, 80, 150], dtype=np.int32)
        self.kernel_size = 5
        self.min_area = 1000
        self.max_area = 25000

        # --- Camera Parameters ---
        self.sizeCamX = 1280
        self.sizeCamY = 720
        self.centerCamX = 642.724365234375
        self.centerCamY = 361.9780578613281
        self.focalX = 915.3003540039062
        self.focalY = 914.0320434570312

        self.camera_matrix = np.array([
            [self.focalX, 0, self.centerCamX],
            [0, self.focalY, self.centerCamY],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((5, 1)) 

        # --- ArUco Params ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.marker_length = 0.13  # 13 cm

        # --- TF Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- Callback Groups ---
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # --- Subscribers ---
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self.colorimagecb,
            10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depthimagecb,
            10,
            callback_group=self.cb_group,
        )

        # --- State ---
        self.detection_complete = False
        self.static_transforms = {} # Key: child_frame_id, Value: (x, y, z, rvec)
        self.frozen_image = None
        
        # We need to find: 3 fruits, 1 fertilizer, 1 landing
        self.required_fruits = 3
        self.found_fertilizer = False
        self.found_landing = False

        # --- Timer ---
        self.create_timer(0.1, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow("perception_view", cv2.WINDOW_NORMAL)

        self.get_logger().info("Perception Node started.")

    def depthimagecb(self, data):
        if self.detection_complete: return
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def colorimagecb(self, data):
        if self.detection_complete: return
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_stamp = data.header.stamp
        except Exception as e:
            self.get_logger().error(f"Failed to convert color image: {e}")

    def detect_bad_fruits(self, rgb_image):
        """
        Detects bad fruits using HSV masking and returns sorted list of centroids.
        """
        bad_fruits = []
        if rgb_image is None:
            return bad_fruits

        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_white, self.upper_white)

        k = max(1, int(self.kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            bad_fruits.append({
                "center": (cX, cY),
                "contour": contour,
                "u": cX  # For sorting
            })
        
        # Sort by 'u' (horizontal position) Left -> Right
        bad_fruits.sort(key=lambda x: x["u"])
        return bad_fruits

    def get_cam_to_base_transform(self):
        try:
            query_time = self.image_stamp if self.image_stamp else rclpy.time.Time()
            return self.tf_buffer.lookup_transform(
                "base_link",
                "camera_link",
                rclpy.time.Time(), # Get latest
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as e:
            # self.get_logger().warn(f"TF Lookup failed: {e}")
            return None

    def publish_tf_from_camera_point(self, x, y, z, child_frame_id, trans, rvec=None):
        """
        Transforms point (x,y,z) in camera frame to base_link and publishes TF.
        """
        
        point_in_camera = PointStamped()
        point_in_camera.header.frame_id = "camera_link"
        point_in_camera.header.stamp = self.get_clock().now().to_msg()
        
        # Boilerplate mapping
        point_in_camera.point.x = z
        point_in_camera.point.y = -x
        point_in_camera.point.z = -y

        try:
            point_in_base = tf2_geometry_msgs.do_transform_point(
                point_in_camera, trans
            )
        except Exception as e:
            self.get_logger().warn(f"Point transform failed: {e}")
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = child_frame_id
        
        t.transform.translation.x = point_in_base.point.x
        t.transform.translation.y = point_in_base.point.y
        t.transform.translation.z = point_in_base.point.z
        
        q = quaternion_from_euler(0, 0, 0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)
        
        # Store for static publishing
        self.static_transforms[child_frame_id] = (x, y, z, rvec)

    def process_image(self):
        # If detection complete, just publish static TFs and show frozen image
        if self.detection_complete:
            trans = self.get_cam_to_base_transform()
            if trans is None: return
            
            for child_frame_id, (x, y, z, rvec) in self.static_transforms.items():
                self.publish_tf_from_camera_point(x, y, z, child_frame_id, trans, rvec)
            
            if SHOW_IMAGE and self.frozen_image is not None:
                cv2.imshow("perception_view", self.frozen_image)
                cv2.waitKey(1)
            return

        if self.cv_image is None or self.depth_image is None or self.image_stamp is None:
            return

        display_image = self.cv_image.copy()
        trans = self.get_cam_to_base_transform()
        if trans is None:
            return

        # ---------------------------------------------------------
        # 1. Bad Fruit Detection
        # ---------------------------------------------------------
        bad_fruits = self.detect_bad_fruits(self.cv_image)
        
        current_fruits_found = 0
        for i, fruit in enumerate(bad_fruits):
            cX, cY = fruit["center"]
            contour = fruit["contour"]

            # Draw
            x_bbox, y_bbox, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_image, (x_bbox, y_bbox), (x_bbox + w, y_bbox + h), (0, 0, 255), 2)
            cv2.putText(display_image, f"BF_{i}", (x_bbox, y_bbox - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Depth
            try:
                raw_depth = self.depth_image[int(cY), int(cX)]
                depth_val = float(raw_depth)
                if np.isnan(depth_val) or depth_val == 0.0:
                    continue
                if depth_val > 10.0:
                    distance_m = depth_val / 1000.0
                else:
                    distance_m = depth_val
            except Exception:
                continue

            # Project to Camera Frame (3D)
            x_cam = (cX - self.centerCamX) * distance_m / self.focalX
            y_cam = (cY - self.centerCamY) * distance_m / self.focalY
            z_cam = distance_m

            # Publish TF
            tf_name = f"{self.team_id}_bad_fruit_{i}"
            self.publish_tf_from_camera_point(x_cam, y_cam, z_cam, tf_name, trans)
            current_fruits_found += 1


        # ---------------------------------------------------------
        # 2. ArUco Detection
        # ---------------------------------------------------------
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        self.found_fertilizer = False
        self.found_landing = False

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
            aruco.drawDetectedMarkers(display_image, corners, ids)

            detected_markers = []
            for i in range(len(ids)):
                c = corners[i][0]
                cX = np.mean(c[:, 0])
                detected_markers.append({
                    'index': i,
                    'id': ids[i][0],
                    'cX': cX,
                    'rvec': rvecs[i][0],
                    'tvec': tvecs[i][0]
                })

            detected_markers.sort(key=lambda x: x['cX'], reverse=True)

            for rank, marker in enumerate(detected_markers):
                if rank == 0:
                    tf_name = "1505_fertiliser_can"
                    self.found_fertilizer = True
                elif rank == 1:
                    tf_name = "landing_ebot"
                    self.found_landing = True
                else:
                    tf_name = f"aruco_{marker['id']}"

                cv2.drawFrameAxes(
                    display_image, self.camera_matrix, self.dist_coeffs, 
                    marker['rvec'], marker['tvec'], 0.1
                )

                tvec = marker['tvec']
                self.publish_tf_from_camera_point(tvec[0], tvec[1], tvec[2], tf_name, trans, marker['rvec'])

        if SHOW_IMAGE:
            cv2.imshow("perception_view", display_image)
            cv2.waitKey(1)
            
        # Check completion condition
        if current_fruits_found >= self.required_fruits and self.found_fertilizer and self.found_landing:
            self.get_logger().info("All objects detected! Freezing perception and switching to static TF.")
            self.detection_complete = True
            self.frozen_image = display_image

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
