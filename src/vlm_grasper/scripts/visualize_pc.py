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
        points.append([point[0], point[1], point[2]])

    if len(points) == 0:
        rospy.logwarn("No valid points received.")
        return

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # Add coordinate frame to the point cloud
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd, mesh_frame])

def main():
    rospy.init_node('point_cloud_visualizer', anonymous=True)
    rospy.Subscriber('/camera/depth/points', PointCloud2, point_cloud_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
