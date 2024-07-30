#!/home/lucfra/miniconda3/envs/ros_env/bin/python

import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sklearn.decomposition import PCA



def process_point_cloud(pcd, voxel_size=0.0065):

    #Only take the biggest cluster
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=50, print_progress=False))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # Select the largest cluster assuming it is the plant
    largest_cluster_indices = np.where(labels == np.argmax(np.bincount(labels[labels >= 0])))[0]
    pcd = pcd.select_by_index(largest_cluster_indices)



    # Downsample the point cloud
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Remove outliers
    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = downpcd.select_by_index(ind)
    
    return inlier_cloud


def pca(pcd, threshold=0.01, filter_points=True):
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


    if filter_points:
        points_without_table = centered_points[np.abs(distances) > threshold]
        return normal_vector, points_without_table
    else:
        return normal_vector, centered_points

        




def normalize_point_cloud(pc, factor=0.3):
    points = np.asarray(pc.points)
    centroid = np.mean(points, axis=0)
    points -= centroid  # Translate points to origin
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points /= (max_dist)  # Scale points to fit within unit sphere
    points *= factor# Scale points to fit within unit sphere
    pc.points = o3d.utility.Vector3dVector(points)
    return pc

def point_cloud_callback(msg, args):

    pc_publisher = args[0]
    # Convert ROS PointCloud2 message to Open3D point cloud
    points = []
    for point in pc2.read_points(msg, skip_nans=True):
        points.append([point[0], point[1], point[2]])

    if not points:
        rospy.logwarn("No points found in the point cloud message.")
        return

    # Convert to numpy array
    points = np.array(points)

    # Create Open3D point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # Get the normal vector of the point cloud
    normal_vector, pc_without_table = pca(pc)

    # Create Open3D point cloud from the filtered points
    pc_without_table_o3d = o3d.geometry.PointCloud()
    pc_without_table_o3d.points = o3d.utility.Vector3dVector(pc_without_table)

    pc_without_table_o3d = process_point_cloud(pc_without_table_o3d)
    pc_without_table_o3d = normalize_point_cloud(pc_without_table_o3d)


    # Visualize the point cloud
    print(f"Normal vector: {normal_vector}")


    rospy.loginfo(f"Publishing processed point cloud with {len(pc_without_table_o3d.points)} points.")
                  

    # Convert Open3D point cloud to ROS PointCloud2 message
    header = msg.header
    pointcloud_msg = pc2.create_cloud_xyz32(header, np.asarray(pc_without_table_o3d.points))

    # Publish the processed point cloud
    pc_publisher.publish(pointcloud_msg)




def main():
    rospy.init_node('point_cloud_transformer', anonymous=True)
    pc_publisher = rospy.Publisher('/point_cloud/processed', PointCloud2, queue_size=10)
    rospy.Subscriber('/point_cloud/unprocessed', PointCloud2, point_cloud_callback, callback_args=(pc_publisher, ))
    
    rospy.spin()




if __name__ == '__main__':
    main()

