import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d

# Load your point cloud data
# Example: point_cloud_data = np.loadtxt("point_cloud.txt")
# If using Open3D to read a point cloud file:
pcd = o3d.io.read_point_cloud("pc_colored_w_table.ply")
points = np.asarray(pcd.points)

print(len(points))

# Center the data
centroid = np.mean(points, axis=0)
centered_points = points - centroid

# Perform PCA
pca = PCA(n_components=3)
pca.fit(centered_points)
normal_vector = pca.components_[2]

# Determine the density of points on each side of the plane
distances = centered_points @ normal_vector
threshold = 0.05
num_positive = np.sum(distances > threshold)
num_negative = np.sum(distances < -threshold)

print(f"Number of points on positive side: {num_positive}")
print(f"Number of points on negative side: {num_negative}")

# Adjust the normal vector to point to the side with more points
if num_negative > num_positive:
    normal_vector = -normal_vector

print(f"Normal vector: {normal_vector}")

# Create a line set to represent the normal vector
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([centroid, centroid + normal_vector]),
    lines=o3d.utility.Vector2iVector([[0, 1]]),
)

# Optionally, set colors for better visualization
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

#Add coordinate frame to the point cloud
# x_axis: red, y_axis: green, z_axis: blue
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# Visualize the point cloud and the normal vector
o3d.visualization.draw_geometries([pcd, line_set, mesh_frame])
