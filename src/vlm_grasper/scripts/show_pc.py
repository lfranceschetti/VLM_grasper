#!/home/lucfra/miniconda3/envs/ros_env/bin/python

import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

def point_cloud_callback(msg):
    # Convert ROS PointCloud2 message to Open3D point cloud
    points = []
    for point in pc2.read_points(msg, skip_nans=True):
        points.append([point[0], point[1], point[2], point[3], point[4], point[5]])

    if not points:
        rospy.logwarn("No points found in the point cloud message.")
        return

    # Convert to numpy array
    points = np.array(points)

    # Extract XYZ and RGB
    xyz = points[:, :3]
    rgb = points[:, 3:]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('point_cloud_listener')

    # Subscribe to the point cloud topic
    rospy.Subscriber('/point_cloud/unprocessed', PointCloud2, point_cloud_callback)

    # Spin to keep the script for exiting
    rospy.spin()
