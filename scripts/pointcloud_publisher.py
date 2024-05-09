#!/usr/bin/env python3
import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os
import rospkg  # Import rospkg to find the package path

def load_and_publish(file_path, publisher):
    cloud = o3d.io.read_triangle_mesh(file_path)
    if not cloud.has_vertex_normals():
        cloud.compute_vertex_normals()
    points = np.asarray(cloud.vertices)
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = file_path
    ros_cloud = pc2.create_cloud_xyz32(header, points)
    publisher.publish(ros_cloud)
    rospy.loginfo("Published point cloud data from file: %s", file_path)

def publisher_node():
    rospy.init_node('pointcloud_publisher')
    pub = rospy.Publisher('/input_point_cloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(0.1)  # 0.1 Hz = 10 seconds

    # Get the path to the package
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('VLM_grasper')  # Replace 'your_package_name' with your actual package name


    ##Graspnet_1B_object
    # base_path = os.path.join(package_path, 'src/VLM_grasper/data_robot/graspnet_1B_object')  # Path to the tests directory
    # files = [f"{base_path}/{str(i).zfill(3)}/convex.obj" for i in range(11)]  # 000 to 010

    #Egad_Eval_Set
    base_path = os.path.join(package_path, 'src/VLM_grasper/data_robot/egad_eval_set')  # Path to the tests directory
    files = [f"{base_path}/A{i}.obj" for i in range(7)]  # A0 to A6

    while not rospy.is_shutdown():
        for file_path in files:
            #Only 003 gave no errors I have yet to figure out why
            if os.path.exists(file_path):
                load_and_publish(file_path, pub)
                rate.sleep()
            else:
                rospy.logwarn("File does not exist: %s", file_path)

if __name__ == '__main__':
    try:
        publisher_node()
    except rospy.ROSInterruptException:
        pass
