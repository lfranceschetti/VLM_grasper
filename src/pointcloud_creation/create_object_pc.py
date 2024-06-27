#!/usr/bin/env python

import rosbag
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
from tqdm import tqdm

def segment_object(color_image, depth_image):
    # Convert color image to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # Use thresholding to create a mask
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)  # Ensure the mask is of type uint8

    # Ensure the mask has the same size as the color image
    if mask.shape != color_image.shape[:2]:
        mask = cv2.resize(mask, (color_image.shape[1], color_image.shape[0]))

    # Apply the mask to the color image
    color_segmented = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Ensure the mask has the same size as the depth image
    if mask.shape != depth_image.shape:
        mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]))

    # Apply the mask to the depth image
    depth_segmented = cv2.bitwise_and(depth_image, depth_image, mask=mask)

    return color_segmented, depth_segmented

def create_point_cloud(color_image, depth_image, camera_info):
    height, width = depth_image.shape
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            z = depth_image[v, u] / 1000.0
            if z == 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append((x, y, z))
            colors.append(color_image[v, u] / 255.0)

    points = np.array(points)
    colors = np.array(colors)
    return points, colors

def main():
    rospy.init_node('bag_to_pointcloud')

    bag = rosbag.Bag('../rosbag/can7.bag')
    bridge = CvBridge()

    color_images = []
    depth_images = []
    camera_info = None

    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw', '/camera/depth/image_rect_raw', '/camera/color/camera_info']):
        if topic == '/camera/color/image_raw':
            cv_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
            color_images.append(cv_img)
        elif topic == '/camera/depth/image_rect_raw':
            depth_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_images.append(depth_img)
        elif topic == '/camera/color/camera_info':
            camera_info = msg

    bag.close()

    if camera_info is None:
        rospy.logerr("Camera info not found in bag file.")
        return

    # Assuming color_images and depth_images are synchronized and of the same length
    points = []
    colors = []
    for color_image, depth_image in tqdm(zip(color_images, depth_images)):
        color_segmented, depth_segmented = segment_object(color_image, depth_image)
        pts, cols = create_point_cloud(color_segmented, depth_segmented, camera_info)
        points.append(pts)
        colors.append(cols)

    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud("output.pcd", pcd)

    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
