import open3d as o3d
import numpy as np

def load_and_segment_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    
    # Filter the point cloud
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # Select the largest cluster assuming it is the plant
    largest_cluster_indices = np.where(labels == np.argmax(np.bincount(labels[labels >= 0])))[0]
    plant_pcd = pcd.select_by_index(largest_cluster_indices)
    
    return plant_pcd

def process_point_cloud(pcd):
    # Downsample the point cloud
    downpcd = pcd.voxel_down_sample(voxel_size=0.005)
    
    # Remove outliers
    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = downpcd.select_by_index(ind)
    
    return inlier_cloud

def upsample_point_cloud(pcd, target_number_of_points):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Get the current number of points and generate more points if needed
    current_num_points = np.asarray(pcd.points).shape[0]
    additional_points = target_number_of_points - current_num_points
    
    if additional_points > 0:
        points = np.asarray(pcd.points)
        indices = np.random.choice(len(points), additional_points, replace=True)
        new_points = points[indices] + np.random.normal(scale=0.005, size=(additional_points, 3))
        all_points = np.vstack((points, new_points))
    else:
        all_points = np.asarray(pcd.points)
    
    upsampled_pcd = o3d.geometry.PointCloud()
    upsampled_pcd.points = o3d.utility.Vector3dVector(all_points)
    upsampled_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.5, 0.5, 0.5]), (len(all_points), 1)))
    
    return upsampled_pcd

def save_view(vis):
    vis.capture_screen_image("plant_view_filled.png")
    print("Image saved to plant_view_filled.png")
    return False

def visualize_point_cloud(pcd):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Optimized Plant Point Cloud")
    vis.add_geometry(pcd)
    vis.register_key_callback(ord(" "), save_view)
    vis.run()
    vis.destroy_window()

def save_point_cloud(pcd, path):
    o3d.io.write_point_cloud(path, pcd)
        

if __name__ == "__main__":
    # Update the path to your point cloud file
    input_ply_path = "pc_colored_w_table.ply"
    
    plant_pcd = load_and_segment_point_cloud(input_ply_path)
    visualize_point_cloud(plant_pcd)
    processed_pcd = process_point_cloud(plant_pcd)
    save_point_cloud(plant_pcd, "pc_processed_w_table.ply")
    visualize_point_cloud(processed_pcd)
