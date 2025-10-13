#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script should be used to implement Task 1B of Krishi coBot (KC) Theme (eYRC 2025-26).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
"""

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task1b_boiler_plate.py
# Functions:
# 			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
# 			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]


import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, Vector3, PointStamped
from tf_transformations import quaternion_from_euler
import tf2_geometry_msgs

# runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False


class FruitsTF(Node):
    """
    ROS2 Boilerplate for fruit detection and TF publishing.
    Students should implement detection logic inside the TODO sections.
    """

    def __init__(self):
        super().__init__("fruits_tf")
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.image_stamp = None
        self.team_id = 1505

        self.lower_white = np.array([0, 0, 65], dtype=np.int32)
        self.upper_white = np.array([180, 80, 150], dtype=np.int32)
        self.kernel_size = 5
        self.min_area = 1000
        self.max_area = 25000

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # callback group handling
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

        # Timer for periodic processing
        self.create_timer(0.05, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow("fruits_tf_view", cv2.WINDOW_NORMAL)

        self.get_logger().info("FruitsTF boilerplate node started.")

    # ---------------- Callbacks ----------------
    def depthimagecb(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_stamp = data.header.stamp
            self.get_logger().info(
                f"Processing new image frame with timestamp: {self.image_stamp}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to convert color image: {e}")

    def bad_fruit_detection(self, rgb_image):
        bad_fruits = []
        if rgb_image is None:
            return bad_fruits

        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_image, self.lower_white, self.upper_white)

        k = max(1, int(self.kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = int(self.min_area)
        max_area = int(self.max_area)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
            extent = float(area) / (bbox_area + 1e-6)

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            fruit_info = {"center": (cX, cY), "contour": contour}
            bad_fruits.append(fruit_info)
        return bad_fruits

    def process_image(self):
        if (
            self.cv_image is None
            or self.depth_image is None
            or self.image_stamp is None
        ):
            return

        display_image = self.cv_image.copy()
        bad_fruits = self.bad_fruit_detection(self.cv_image)

        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312

        bad_fruit_id = 1
        for fruit in bad_fruits:
            cX, cY = fruit["center"]
            contour = fruit["contour"]

            x_bbox, y_bbox, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                display_image,
                (x_bbox, y_bbox),
                (x_bbox + w, y_bbox + h),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                display_image,
                "bad_fruit",
                (x_bbox, y_bbox - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            try:
                raw_depth = self.depth_image[int(cY), int(cX)]
                if raw_depth is None:
                    continue
                depth_val = float(raw_depth)
                if np.isnan(depth_val) or depth_val == 0.0:
                    continue
                # need to probably change not a good way
                if depth_val > 10.0:
                    distance_m = depth_val / 1000.0
                else:
                    distance_m = depth_val
            except IndexError:
                continue
            except Exception as e:
                self.get_logger().warn(f"Depth read error at ({cX},{cY}): {e}")
                continue

            x = (cX - centerCamX) * distance_m / focalX
            y = (cY - centerCamY) * distance_m / focalY
            z = distance_m

            cam_to_base_transform = None
            try:
                query_time = (
                    self.image_stamp
                    if self.image_stamp is not None
                    else rclpy.time.Time()
                )
                cam_to_base_transform = self.tf_buffer.lookup_transform(
                    "base_link",
                    "camera_link",
                    query_time,
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
            except Exception as e1:
                try:
                    cam_to_base_transform = self.tf_buffer.lookup_transform(
                        "base_link",
                        "camera_link",
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.2),
                    )
                except Exception as e2:
                    self.get_logger().warn(
                        f"Could not lookup transform base_link<-camera_link (ts then latest): {e1} / {e2}"
                    )
                    cam_to_base_transform = None

            if cam_to_base_transform is None:
                continue

            point_in_camera = PointStamped()
            point_in_camera.header.frame_id = "camera_link"
            point_in_camera.header.stamp = self.get_clock().now().to_msg()
            point_in_camera.point.x = z
            point_in_camera.point.y = -x
            point_in_camera.point.z = -y

            try:
                point_in_base = tf2_geometry_msgs.do_transform_point(
                    point_in_camera, cam_to_base_transform
                )

                t_base_fruit = TransformStamped()
                t_base_fruit.header.stamp = self.get_clock().now().to_msg()
                t_base_fruit.header.frame_id = "base_link"
                t_base_fruit.child_frame_id = f"{self.team_id}_bad_fruit_{bad_fruit_id}"
                t_base_fruit.transform.translation.x = point_in_base.point.x
                t_base_fruit.transform.translation.y = point_in_base.point.y
                t_base_fruit.transform.translation.z = point_in_base.point.z
                q = quaternion_from_euler(0, 0, 0)
                t_base_fruit.transform.rotation.x = q[0]
                t_base_fruit.transform.rotation.y = q[1]
                t_base_fruit.transform.rotation.z = q[2]
                t_base_fruit.transform.rotation.w = q[3]
                self.tf_broadcaster.sendTransform(t_base_fruit)
            except Exception as e:
                self.get_logger().warn(
                    f"Could not transform point to base_link or publish TF: {e}"
                )

            bad_fruit_id += 1

        if SHOW_IMAGE:
            cv2.imshow("fruits_tf_view", display_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
