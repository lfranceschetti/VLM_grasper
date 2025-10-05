#!/usr/bin/env python

import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sklearn.decomposition import PCA
from geometry_msgs.msg import Vector3Stamped
import time
import os
import roslib



def limit_distance(pcd):

    distance = rospy.get_param("max_distance_points", 1.8)
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points, axis=1)
    mask = distances < distance
    pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd



def take_object_clusters(pcd, n=1):
    # Cluster the point cloud
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=50, print_progress=False))
    unique_labels = np.unique(labels)

    # Filter out noise (label -1)
    valid_labels = unique_labels[unique_labels >= 0]

    # Find the largest n clusters
    largest_clusters_indices = []
    for i in range(min(n, len(valid_labels))):
        # Find the largest cluster by counting points in each valid cluster
        largest_cluster_label = valid_labels[np.argmax([np.sum(labels == label) for label in valid_labels])]
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        largest_clusters_indices.extend(largest_cluster_indices)

        # Remove the largest cluster from further consideration
        valid_labels = valid_labels[valid_labels != largest_cluster_label]

    # Select the points that belong to the largest n clusters
    pcd = pcd.select_by_index(largest_clusters_indices)

    

    return pcd



def pca(pcd, threshold=0.012, filter_points=True):
    points = np.asarray(pcd.points)

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    pca = PCA(n_components=3)
    pca.fit(centered_points)
    normal_vector = pca.components_[2]

    distances = centered_points @ normal_vector
    num_positive = np.sum(distances > threshold)
    num_negative = np.sum(distances < -threshold)

    if num_negative > num_positive:
        normal_vector = -normal_vector

    pc = o3d.geometry.PointCloud()

    if filter_points:
        points_without_table = points[np.abs(distances) > threshold]
        pc.points = o3d.utility.Vector3dVector(points_without_table)
    else:
        pc.points = o3d.utility.Vector3dVector(points)

        
    return normal_vector, pc


def ransac_plane_segmentation(pcd):

    threshold = rospy.get_param("ransac_threshold", 0.0123)

    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=1000)

    # Extract inliers and outliers
    table_cloud = pcd.select_by_index(inliers)
    objects_cloud = pcd.select_by_index(inliers, invert=True)

    normal_vector = np.array(plane_model[:3])  
    
    points = np.asarray(pcd.points)
    distances = points @ normal_vector + plane_model[3]
    num_positive = np.sum(distances > threshold)
    num_negative = np.sum(distances < -threshold)

    if num_negative > num_positive:
        normal_vector = -normal_vector

    # Normal vector from the plane coefficients

    return normal_vector, objects_cloud


def point_cloud_callback(msg, args):

    if rospy.get_param('point_cloud_transformed', False):
        return
    
    now = time.time() 
    #Save start time in milliseconds
    rospy.set_param('t_start',  now)

    rospy.loginfo("Processing point cloud...")

    pc_publisher = args[0]
    pc_objects_publisher = args[1]
    normal_vector_publisher = args[2]
    center_of_pc_pub = args[3]
    # Convert ROS PointCloud2 message to Open3D point cloud
    points = []
    for point in pc2.read_points(msg, skip_nans=True):
        points.append([point[0], point[1], point[2]])

    if not points:
        rospy.logwarn("No points found in the point cloud message.")
        return

    # Convert to numpy array
    points = np.array(points)



    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    pc = limit_distance(pc)


    # Get the normal vector of the point cloud
    normal_vector, pc = ransac_plane_segmentation(pc)

    #Create normal vector message
    normal_vector_msg = Vector3Stamped()
    normal_vector_msg.header = msg.header
    normal_vector_msg.vector.x = normal_vector[0]
    normal_vector_msg.vector.y = normal_vector[1]
    normal_vector_msg.vector.z = normal_vector[2]
    normal_vector_publisher.publish(normal_vector_msg)

    n = rospy.get_param("number_of_objects", 1)

    pc_objects = take_object_clusters(pc, n=n)

    #Visualize the point cloud
    # o3d.visualization.draw_geometries([pc_objects])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pc_objects)
    vis.poll_events()
    vis.update_renderer()

    results_path = os.path.join(roslib.packages.get_pkg_dir('vlm_grasper'), "experiments", rospy.get_param('scene', 'unknown'), "results") 

    vis.capture_screen_image(os.path.join(results_path, "point_cloud.png"))


    voxel_size = rospy.get_param("voxel_downsampling", 0.005)
    # Downsample the point cloud
    if voxel_size > 0.0:
        pc_objects = pc_objects.voxel_down_sample(voxel_size=voxel_size)

    # Remove outliers
    cl, ind = pc_objects.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc_objects = pc_objects.select_by_index(ind)

    # Create a point cloud with the center of the point cloud
    center_of_pc = np.mean(np.asarray(pc_objects.points), axis=0)
    center_of_pc_msg = Vector3Stamped()
    center_of_pc_msg.header = msg.header
    center_of_pc_msg.vector.x = center_of_pc[0]
    center_of_pc_msg.vector.y = center_of_pc[1]
    center_of_pc_msg.vector.z = center_of_pc[2]
    center_of_pc_pub.publish(center_of_pc_msg)


    pc_objects_msg = pc2.create_cloud_xyz32(msg.header, np.asarray(pc_objects.points))
    pc_objects_publisher.publish(pc_objects_msg)

    # Convert Open3D point cloud to ROS PointCloud2 message
    header = msg.header

    pointcloud_msg = pc2.create_cloud_xyz32(header, np.asarray(pc_objects.points))

    # Publish the processed point cloud
    now = time.time() 
    rospy.set_param('t_pc_processed', now)

    rospy.loginfo("Publishing processed point cloud.")
    pc_publisher.publish(pointcloud_msg)



    # rospy.signal_shutdown("Message published, shutting down node.")





def main():
    rospy.init_node('pointcloud_transformer', anonymous=True)


    pc_publisher = rospy.Publisher('/point_cloud/processed', PointCloud2, queue_size=10)
    pc_objects_publisher = rospy.Publisher('/point_cloud/objects', PointCloud2, queue_size=10)
    normal_vector_publisher = rospy.Publisher('/surface_normal', Vector3Stamped, queue_size=10)
    center_of_pc_pub = rospy.Publisher('/center_of_pc', Vector3Stamped, queue_size=10)

    rospy.Subscriber('/camera/depth/color/points', PointCloud2, point_cloud_callback, callback_args=(pc_publisher, pc_objects_publisher, normal_vector_publisher, center_of_pc_pub))

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        





if __name__ == '__main__':
    main()

