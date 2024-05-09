import open3d as o3d
import numpy as np
import torch
from catkin_ws.src.VLM_grasper.utils.perform_edge_grasp import edge_grasp

# Assuming preprocess_and_predict is defined as you've provided and is ready to use
def load_obj_as_pointcloud(filepath):
    # Load an OBJ file into an Open3D point cloud object
    mesh = o3d.io.read_triangle_mesh(filepath)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    pc = mesh.sample_points_poisson_disk(number_of_points=2048)  # Adjust the number of points as needed
    return pc

# Example usage
if __name__ == "__main__":
    filepath = "tests/000/convex.obj"
    point_cloud = load_obj_as_pointcloud(filepath)
    result = edge_grasp(point_cloud, add_noise=True, plotting=False)
    print(result)