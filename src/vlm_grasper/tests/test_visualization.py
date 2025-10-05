import numpy as np
import matplotlib.pyplot as plt
from vlm_grasper.utils.visualization_utils import visualize_grasps
import rospy
import open3d as o3d
import os
import rospkg

LENGTH = 0.09

def create_transformation_matrix(position, approach, normal):
    # Convert attributes to arrays
    position = np.array([position.x, position.y, position.z])
    approach = np.array([approach.x, approach.y, approach.z])
    normal = np.array([normal.x, normal.y, normal.z])


    z_axis = approach
    #Make that the normal direction is flipped and shows in the other direction
    normal = -normal
    x_axis = normal
    y_axis = np.cross(x_axis, z_axis)  # Compute the missing Y-axis

    # Normalize the axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)


    # Create rotation matrix
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Create translation vector
    translation_vector = position
    # We need to move the gripper back by the length of the gripper
    translation_vector += LENGTH * approach
    


    # Combine into a 4x4 transformation matrix
    transformation_matrix = np.eye(4)

    rotation_matrix = np.dot(np.array([[-1,0,0], [0, -1, 0], [0, 0, -1]]), rotation_matrix)

    transformation_matrix[:3, :3] = rotation_matrix    


    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix

def visualize_grasps_data(grasps, points):
    transformation_matrices = []
    scores = []

    for grasp in grasps:
        print(f"Grasp: {grasp}")

        # Corrected dictionary access
        position = grasp['position']
        approach = grasp['approach']
        normal = grasp['normal']
        score = grasp['score']


        # Since position, approach, and normal seem to be stored as dictionaries themselves, you should correct here as well:
        matrix = create_transformation_matrix(
            position=position,
            approach=approach,
            normal=normal
        )
        transformation_matrices.append(matrix)

    scores = np.array([grasp['score'] for grasp in grasps])
    transformation_matrices = np.array(transformation_matrices)

    visualize_grasps(points, {0: transformation_matrices}, {0: scores}, plot_opencv_cam=True, pc_colors=None)




if __name__ == '__main__':
    
    #Get vlm_grasper package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vlm_grasper')



    #Get the grasps from grasps.npy
    grasps = np.load(os.path.join(package_path, 'grasps.npy'), allow_pickle=True)

    #Get the point cloud from pointcloud.ply
    pcd = o3d.io.read_point_cloud(os.path.join(package_path, 'pointcloud.ply'))
    points = np.asarray(pcd.points)



    visualize_grasps_data(grasps, points)
