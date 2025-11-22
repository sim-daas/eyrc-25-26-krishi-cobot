#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script implements ArUco marker detection and TF publishing.
*  Based on task1b_boiler_plate.py
*
*****************************************************************************************
"""

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
from tf_transformations import quaternion_from_matrix, quaternion_from_euler
import tf2_geometry_msgs

# Runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = True

class ArucoTF(Node):
    """
    ROS2 Node for ArUco marker detection and TF publishing.
    """

    def __init__(self):
        super().__init__("aruco_tf_node")
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.image_stamp = None
        self.team_id = 1505

        # Camera Parameters (from task1b_boiler_plate.py)
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
        
        self.dist_coeffs = np.zeros((5, 1)) # Assuming no distortion as per simulation

        # ArUco Configuration
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.marker_length = 0.13  # 13 cm as per boilerplate info

        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Callback groups
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
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

        # Timer
        self.create_timer(0.1, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow("aruco_tf_view", cv2.WINDOW_NORMAL)

        self.get_logger().info("ArucoTF node started.")

    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_stamp = data.header.stamp
        except Exception as e:
            self.get_logger().error(f"Failed to convert color image: {e}")

    def process_image(self):
        if self.cv_image is None or self.image_stamp is None:
            return

        # Copy image for display
        display_image = self.cv_image.copy()
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        # Detect Markers
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None:
            # Estimate Pose for all markers
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

            # Draw markers
            aruco.drawDetectedMarkers(display_image, corners, ids)

            # Collect detected markers with their centers for sorting
            detected_markers = []
            for i in range(len(ids)):
                # Calculate center X coordinate
                # corners[i] is shape (1, 4, 2)
                c = corners[i][0]
                cX = np.mean(c[:, 0])
                
                detected_markers.append({
                    'index': i,
                    'id': ids[i][0],
                    'cX': cX,
                    'rvec': rvecs[i][0],
                    'tvec': tvecs[i][0]
                })

            # Sort markers by cX in descending order (Right -> Left)
            detected_markers.sort(key=lambda x: x['cX'], reverse=True)

            # Assign names based on position
            for rank, marker in enumerate(detected_markers):
                if rank == 0:
                    # The right-most marker
                    tf_name = "1505_fertilizer_can"
                elif rank == 1:
                    # The second right-most (the "other" one)
                    tf_name = "landing_ebot"
                else:
                    # Any additional markers
                    tf_name = f"aruco_{marker['id']}"

                # Draw axis
                cv2.drawFrameAxes(
                    display_image, self.camera_matrix, self.dist_coeffs, 
                    marker['rvec'], marker['tvec'], 0.1
                )

                # Get position from tvec (in camera frame)
                tvec = marker['tvec']
                rvec = marker['rvec']

                z_cam = tvec[2]
                x_cam = tvec[0]
                y_cam = tvec[1]

                # Transform to base_link and publish
                self.publish_tf(x_cam, y_cam, z_cam, rvec, tf_name)

        if SHOW_IMAGE:
            cv2.imshow("aruco_tf_view", display_image)
            cv2.waitKey(1)

    def publish_tf(self, x, y, z, rvec, child_frame_id):
        # 1. Get Transform from base_link to camera_link
        try:
            query_time = self.image_stamp if self.image_stamp else rclpy.time.Time()
            
            trans = self.tf_buffer.lookup_transform(
                "base_link",
                "camera_link",
                rclpy.time.Time(), # Get latest available
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as e:
            self.get_logger().warn(f"Could not lookup transform: {e}")
            return

        # 2. Create a PoseStamped/PointStamped for the marker in Camera Frame
        point_in_camera = PointStamped()
        point_in_camera.header.frame_id = "camera_link"
        point_in_camera.header.stamp = self.get_clock().now().to_msg()
        
        # Apply boilerplate transformation (swapping axes)
        point_in_camera.point.x = z
        point_in_camera.point.y = -x
        point_in_camera.point.z = -y

        # 3. Transform point to Base Link
        try:
            point_in_base = tf2_geometry_msgs.do_transform_point(
                point_in_camera, trans
            )
        except Exception as e:
            self.get_logger().warn(f"Transform error: {e}")
            return

        # 4. Publish TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = child_frame_id
        
        t.transform.translation.x = point_in_base.point.x
        t.transform.translation.y = point_in_base.point.y
        t.transform.translation.z = point_in_base.point.z
        
        # Identity rotation as per boilerplate
        q = quaternion_from_euler(0, 0, 0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoTF()
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
