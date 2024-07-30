import numpy as np
import open3d as o3d
import torch
import rospy
from vlm_grasper.utils.perform_edge_grasp import perform_edge_grasp
import os
import sys
from vlm_grasper.utils.visualization_utils_o3d import visualize_grasps



LENGTH = 0.09

def normalize_point_cloud(pc):
    points = np.asarray(pc.points)
    centroid = np.mean(points, axis=0)
    points -= centroid  # Translate points to origin
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points /= (max_dist)  # Scale points to fit within unit sphere
    points *= 0.1 # Scale points to fit within unit sphere
    pc.points = o3d.utility.Vector3dVector(points)
    return pc


def find_grasp_positions(input_file, output_path):


    #Load the point cloud from the .ply file 
    pc = o3d.io.read_point_cloud(input_file)

    pc = normalize_point_cloud(pc)

    # If points are to close to each other, the grasp detection will fail
    # So we need to downsample the point cloud
    pc = pc.voxel_down_sample(voxel_size=0.0065)

    # Normalize the point cloud
    print(pc)

    # grasp_data = perform_edge_grasp(pc, add_noise=True, plotting=False)

    # if grasp_data is None:
    #     rospy.logwarn("No grasps found")
    #     return

    # score, depth_projection, approaches, sample_pos, des_normals = grasp_data


    # # FOR NOW TAKE THE 10 WITH THE HIGHEST SCORE

    # # Ensure each step is explicitly copied if necessary
    # score = score.cpu().detach().numpy()
    # #Take the indices of the 10 highest scores
    # sorted_indices = np.argsort(score)[::-1]
  
    # best_indices = sorted_indices[:10]
    # scores = score[best_indices].copy()
    # sample_pos = sample_pos.cpu().numpy()[best_indices].copy()  # Force a copy here
    # des_normals = des_normals.cpu().numpy()[best_indices].copy()
    # depth_projection = depth_projection.cpu().numpy()[best_indices].copy()
    # approaches = approaches.cpu().numpy()[best_indices].copy()  

    sample_pos = [[-0.05305453,0.09923805,0.06577206]]
    approaches = [[-0.02154241,0.9678095, 0.25076035]]
    des_normals = [[-0.991959, -0.05197844,0.11539322]]

    

    

    sample_pos = np.array(sample_pos)
    approaches = np.array(approaches)
    des_normals = np.array(des_normals)

    scores = [0.9]


    transformation_matrices = []
    for i in range(len(sample_pos)):
        print("sample_pos[i]: ", sample_pos[i])
        print("approaches[i]: ", approaches[i])
        print("des_normals[i]: ", des_normals[i])
        matrix = create_transformation_matrix(sample_pos[i], approaches[i], des_normals[i])
        transformation_matrices.append(matrix)

    
    
    transformation_matrices = np.array(transformation_matrices)
    scores = np.array(scores)

    scores = {0: scores}
    transformation_matrices = {0: transformation_matrices}

    points = np.array(pc.points)

    print("points: ", points)
    print("transformation_matrices: ", transformation_matrices)

    visualize_grasps(points, transformation_matrices, scores, pc_colors=None)












def create_transformation_matrix(position, approach, normal):
    # Convert attributes to arrays


    z_axis = approach
    #Make that the normal direction is flipped and shows in the other direction
    x_axis = -normal
    y_axis = np.cross(x_axis, z_axis)  # Compute the missing Y-axis

    # Normalize the axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)


    # Create rotation matrix (For some reason, the axes are flipped)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    rotation_matrix = np.dot(np.array([[-1,0,0], [0, -1, 0], [0, 0, -1]]), rotation_matrix)

    # Create translation vector 
    translation_vector = position
    # We need to move the gripper back by the length of the gripper
    translation_vector += LENGTH * approach
    
    #Add an additonal rotation around the x-axis to make the gripper face the object
    rotation_matrix = np.dot(rotation_matrix, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    #Move the gripper up
    translation_vector += 0.17 * np.array([0.28, -0.8, -1])

    # Combine into a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix    
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


if __name__ == '__main__':

    input_file = 'plant_processed.ply'
    output_path = 'grasps.png'

    find_grasp_positions(input_file, output_path)
