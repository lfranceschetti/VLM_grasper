import cv2


import numpy as np
import open3d as o3d
import torch
import rospy
from vlm_grasper.utils.perform_edge_grasp import perform_edge_grasp
import os
import sys
from vlm_grasper.utils.visualization_utils_o3d import visualize_grasps
from grasp_visualizations.grasp_pointcloud import get_grasp_pointclouds
from camera_info import get_camera_info


# Can be found with pca.py
NORMAL_SURFACE = np.array([ 0.04256444, -0.48665496, -0.87255671])
original_image = cv2.imread('image_1720013430301606560.png')
depth_image = np.load('depth_image_1720013430300073603.npy')

LENGTH = 0.09

def normalize_point_cloud(pc):
    points = np.asarray(pc.points)
    centroid = np.mean(points, axis=0)
    points -= centroid  # Translate points to origin
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points /= (max_dist)  # Scale points to fit within unit sphere
    points *= 0.3# Scale points to fit within unit sphere
    pc.points = o3d.utility.Vector3dVector(points)
    return pc

K_d, D_d = get_camera_info("depth")

def backproject_point_to_depth_image(point, depth_image, K_d):
    """
    Backproject a point to the depth image.

    Arguments:
        point {np.ndarray} -- 3D point (X, Y, Z)
        depth_image {np.ndarray} -- Depth image
        K_d {np.ndarray} -- Camera intrinsic matrix

    Returns:
        np.ndarray -- 2D point in the depth image and depth value
    """
    # Convert the 3D point to homogeneous coordinates
    point_homog = np.append(point, 1)
    
    # Project the point to the 2D image plane using the intrinsic matrix
    point_image_homog = np.dot(K_d, point_homog[:3])
    
    # Normalize to get the 2D image coordinates
    u = point_image_homog[0] / point_image_homog[2]
    v = point_image_homog[1] / point_image_homog[2]
    
    # Convert to integer pixel coordinates
    u = int(u)
    v = int(v)
    
    # Check if the projected point is within the image bounds
    if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
        # Get the depth value from the z-coordinate of the point
        depth = point[2] * 1000  # Convert to millimeters
        return np.array([u, v]), depth
    else:
        # Return None if the point is out of bounds
        return None, None




def find_grasp_positions(input_file, output_path):


    #Load the point cloud from the .ply file 
    pc = o3d.io.read_point_cloud(input_file)

    # pc = normalize_point_cloud(pc)

    # If points are to close to each other, the grasp detection will fail
    # So we need to downsample the point cloud
    pc = pc.voxel_down_sample(voxel_size=0.0065)

    # Normalize the point cloud
    print(pc)

    grasp_data = perform_edge_grasp(pc, add_noise=True, plotting=False)

    if grasp_data is None:
        rospy.logwarn("No grasps found")
        return

    score, depth_projection, approaches, sample_pos, des_normals = grasp_data

    #Filter out the grasps where approah * NORMAL_SURFACE < 0
    #This means that the gripper is pointing in the wrong direction
    approaches_numpy = approaches.cpu().numpy()
    print("APPROACHES: ", approaches_numpy.shape)
    print("NORMAL_SURFACE: ", NORMAL_SURFACE.shape)
    # possible_indices = [i for i, approach in enumerate(approaches.cpu().numpy()) if np.dot(approach, NORMAL_SURFACE) > 0]
    possible_indices = [i for i, approach in enumerate(approaches.cpu().numpy())]

    # # FOR NOW TAKE THE 10 WITH THE HIGHEST SCORE

    # Ensure each step is explicitly copied if necessary
    score = score.cpu().detach().numpy()
    #Take the indices of the 10 highest scores
    sorted_indices = np.argsort(score)[::-1]
  
    best_indices = [i for i in sorted_indices if i in possible_indices][:5]
    scores = score[best_indices].copy()
    sample_pos = sample_pos.cpu().numpy()[best_indices].copy()  # Force a copy here
    des_normals = des_normals.cpu().numpy()[best_indices].copy()
    approaches = approaches.cpu().numpy()[best_indices].copy()  
    

    sample_pos = np.array(sample_pos)
    approaches = np.array(approaches)
    des_normals = np.array(des_normals)


    transformation_matrices = []
    for i in range(len(sample_pos)):
        matrix = create_transformation_matrix(sample_pos[i], approaches[i], des_normals[i])
        transformation_matrices.append(matrix)

    
    transformation_matrices = np.array(transformation_matrices)
    scores = np.array(scores)

    scores = {0: scores}
    transformation_matrices = {0: transformation_matrices}


    grasp_pointclouds = get_grasp_pointclouds(transformation_matrices, scores)

    print("Grasp point clouds: ", grasp_pointclouds)

    # o3d.visualization.draw_geometries([pc] + grasp_pointclouds)

    

    #For each point in grasp_pointclouds, backproject it to the depth image
    #If the point is closer to the camera than the current depth value, update the depth value
    #If the point is closer to the camera than the current depth value, change the color of the pixel in the color image to the color of the point
    #If the point is further away, ignore it
    #Save the color image as a .png file
    o3d.visualization.draw_geometries([pc] + grasp_pointclouds)

    for grasp_pointcloud in grasp_pointclouds:
        points = np.asarray(grasp_pointcloud.points)
        for point in points:
            
            #Backproject the point to the depth image
            pixel, depth = backproject_point_to_depth_image(point, depth_image, K_d)

            print("PIXEL: ", pixel)
            print("DEPTH: ", depth)
            print("ORIGINAL DEPTH: ", depth_image[pixel[1], pixel[0]])
            if pixel is not None and depth is not None:
                #If the point is closer to the camera than the current depth value, update the depth value
                if depth < depth_image[pixel[1], pixel[0]]:
                    print("UPDATING DEPTH")
                    depth_image[pixel[1], pixel[0]] = depth
                    #Change the color of the pixel in the color image to the color of the point
                    original_image[pixel[1], pixel[0]] = point[:3] * 255

    # cv2.imwrite(output_path, original_image)

            













def create_transformation_matrix(position, approach, normal):
    # Convert attributes to arrays


    z_axis = -approach
    #Make that the normal direction is flipped and shows in the other direction
    x_axis = -normal
    y_axis = np.cross(x_axis, z_axis)  # Compute the missing Y-axis

    # Normalize the axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # # Create rotation matrix (For some reason, the axes are flipped)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Create translation vector 
    translation_vector = position
    # We need to move the gripper back by the length of the gripper
    translation_vector += LENGTH * approach
    

    # Combine into a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix    
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


if __name__ == '__main__':

    input_file = 'pc_processed.ply'
    output_path = 'grasps.png'

    find_grasp_positions(input_file, output_path)
