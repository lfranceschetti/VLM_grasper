#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg

# Constants
MAX_DEPTH = 1000

# Initialize ROS node
rospy.init_node('image_to_pc')

# CvBridge instance
bridge = CvBridge()

# Global variables to hold camera intrinsics
K_c = None
D_c = None
K_d = None
D_d = None

# Global variables for point cloud and color images
latest_color_image = None
latest_depth_image = None
last_publish_time = rospy.Time.now()

# Point cloud publisher
point_cloud_pub = rospy.Publisher('/point_cloud/unprocessed', PointCloud2, queue_size=10)

def camera_info_callback(data, camera_type):
    global K_c, D_c, K_d, D_d
    if camera_type == 'color':
        K_c = np.array(data.K).reshape(3, 3)
        D_c = np.array(data.D)
    elif camera_type == 'depth':
        K_d = np.array(data.K).reshape(3, 3)
        D_d = np.array(data.D)

def depth_to_point_cloud(depth):
    """Transform a depth image into a point cloud."""
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth < MAX_DEPTH) & (depth > 0)

    z = np.where(valid, depth.astype(np.float32) / 1000.0, np.nan)  # Convert depth to meters if in millimeters
    x = np.where(valid, z * (c - K_d[0, 2]) / K_d[0, 0], 0)
    y = np.where(valid, z * (r - K_d[1, 2]) / K_d[1, 1], 0)
    return np.dstack((x, y, z))

def add_color_to_point_cloud(point_cloud, color_image):
    """Add correct color to a point cloud using the depth to color transformation."""
    rows, cols, _ = point_cloud.shape
    colors = np.zeros((rows, cols, 3), dtype=np.float32)
    for v_d in range(rows):
        for u_d in range(cols):
            z = point_cloud[v_d, u_d, 2]
            if np.isfinite(z) and z < MAX_DEPTH and z > 0:
                colors[v_d, u_d] = color_image[v_d, u_d] / 255.0
    return colors

def callback(color_msg, depth_msg):
    global K_c, D_c, K_d, D_d, latest_color_image, latest_depth_image, msg_time

    if K_c is None or K_d is None:
        rospy.logwarn("Camera intrinsics not yet received.")
        return

    try:
        color_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, "16UC1")
    except CvBridgeError as e:
        rospy.logerr("CvBridge error: {}".format(e))
        return

    latest_color_image = color_image
    latest_depth_image = depth_image

    msg_time = depth_msg.header.stamp

def publish_point_cloud(event):
    global latest_color_image, latest_depth_image, last_publish_time
    if latest_color_image is None or latest_depth_image is None:
        rospy.logwarn("Images not yet received.")
        return
    
    # rospy.loginfo(f"Latest depth image shape: {latest_depth_image.shape}")

    point_cloud = depth_to_point_cloud(latest_depth_image)
    valid_mask = np.isfinite(point_cloud[..., 2])
    colors = add_color_to_point_cloud(point_cloud, latest_color_image)
    point_cloud = point_cloud[valid_mask].reshape(-1, 3)

    # Create a PointCloud2 message
    header = std_msgs.msg.Header()
    header.stamp = msg_time
    header.frame_id = "camera_link"
    points = np.hstack((point_cloud, colors[valid_mask].reshape(-1, 3)))
    cloud_msg = pc2.create_cloud(header, fields=[
        pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
        pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
        pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
        pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
        pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
        pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1),
    ], points=points)

    rospy.loginfo(f"Publishing unprocessed point cloud with {len(point_cloud)} points.")
    #Show with open3d

    

    point_cloud_pub.publish(cloud_msg)

if __name__ == '__main__':

    # Subscribe to the camera info topics
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_callback, 'color')
    rospy.Subscriber('/camera/depth/camera_info', CameraInfo, camera_info_callback, 'depth')

    # Synchronize the color and depth images
    color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
    ts.registerCallback(callback)

    # Publish point cloud every 5 seconds
    rospy.Timer(rospy.Duration(5), publish_point_cloud)

    rospy.spin()
