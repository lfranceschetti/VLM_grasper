import cv2
import numpy as np
import open3d as o3d
import rosbag
import tf2_ros
import tf.transformations as tr
from camera_info import get_camera_info
from cv_bridge import CvBridge
import rospy
from tf2_ros import TransformBroadcaster# Initialize ROS node
import time
import threading


# Load the original image and depth image
original_image = cv2.imread('image_1720013430301606560.png')
depth_image = np.load('depth_image_1720013430300073603.npy')
segmentation_mask = np.load('image_1720013430301606560_mask.npy')


K_c, D_c = get_camera_info("color")
K_d, D_d = get_camera_info("depth")


#Only take points that are less than 1 meter away
MAX_DEPTH = 1000

print(f"Depth image shape: {depth_image.shape}")
print(f"Original image shape: {original_image.shape}")

def depth_to_point_cloud(depth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centered at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depth values. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.
    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    #Apply the mask to the depth image and only take the ones with depth < MAX_DEPTH
    # valid = np.where((depth < MAX_DEPTH) & (depth > 0)) and segmentation_mask
    valid = (depth < MAX_DEPTH) & (depth > 0) 

    z = np.where(valid, depth.astype(np.float32) / 1000.0, np.nan)  # Convert depth to meters if in millimeters
    x = np.where(valid, z * (c - K_d[0, 2]) / K_d[0, 0], 0)
    y = np.where(valid, z * (r - K_d[1, 2]) / K_d[1, 1], 0)
    return np.dstack((x, y, z))


def add_color_to_point_cloud(point_cloud, color_image):
    """Add correct color to a point cloud using the depth to color transformation."""
    rows, cols, _ = point_cloud.shape
    colors = np.zeros((rows, cols, 3), dtype=np.float32)
    new_image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for v_d in range(rows):
        for u_d in range(cols):
            z = point_cloud[v_d, u_d, 2] 
            if np.isfinite(z) and z < MAX_DEPTH and z > 0:
                colors[v_d, u_d] = color_image[v_d, u_d] / 255.0
                new_image[v_d, u_d] = color_image[v_d, u_d]

    cv2.imwrite("new_image2.png", new_image)
    return colors

point_cloud = depth_to_point_cloud(depth_image)

valid_mask = np.isfinite(point_cloud[..., 2])

colors= add_color_to_point_cloud(point_cloud, original_image)

point_cloud = point_cloud[valid_mask].reshape(-1, 3)

# Create an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd.colors = o3d.utility.Vector3dVector(colors[valid_mask].reshape(-1, 3))

# Visualize the colored point cloud
# o3d.visualization.draw_geometries([pcd])

#Add coordinate frame to the point cloud
# x_axis: red, y_axis: green, z_axis: blue
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd, mesh_frame])



# Save the point cloud to a .ply file
o3d.io.write_point_cloud("pc_colored_w_table.ply", pcd)