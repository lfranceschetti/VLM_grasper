#!/usr/bin/env python
import rospy
import numpy as np
import open3d as o3d
import vlm_grasper.utils.mesh_utils as mesh_utils



LENGTH = 0.09


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


def create_transformation_matrix(position, approach, normal):
    # Convert attributes to arrays
    position = np.array([position.x, position.y, position.z])
    approach = np.array([approach.x, approach.y, approach.z])
    normal = np.array([normal.x, normal.y, normal.z])


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
    translation_vector +=  LENGTH * approach
    

    # Combine into a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix    
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix





def get_grasp_pointclouds(grasp_matrices, scores, gripper_width=0.08):
    grasp_pointclouds = []


    for i, grasp_matrix in enumerate(grasp_matrices):
        gripper_opening = np.ones(len(grasp_matrix)+1) * gripper_width
        color = get_grasp_color(i)

        grasp_pointcloud = draw_grasp(grasp_matrix, np.eye(4), gripper_opening, color)
        grasp_pointclouds.append(grasp_pointcloud)


    return grasp_pointclouds


def generate_points_along_line(start, end, num_points=100):

    return np.linspace(start, end, num_points)





def draw_grasp(grasp, cam_pose, gripper_opening, color):
    """
    Draws wireframe grasps from given camera pose and with given gripper openings

    Arguments:
        grasps {np.ndarray} -- Nx4x4 grasp pose transformations
        cam_pose {np.ndarray} -- 4x4 camera pose transformation
        gripper_openings {np.ndarray} -- Nx1 gripper openings

    Keyword Arguments:
        color {tuple} -- color of all grasps (default: {(0,1.,0)})
        colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
        tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
    """
    num_points_per_line = 100   

    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    all_points = []
    
    gripper_control_points_closed = grasp_line_plot.copy()
    gripper_control_points_closed[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * gripper_opening / 2
    
    pts = np.matmul(gripper_control_points_closed, grasp[:3, :3].T)
    pts += np.expand_dims(grasp[:3, 3], 0)
    pts_homog = np.concatenate((pts, np.ones((7, 1))), axis=1)
    pts = np.dot(pts_homog, cam_pose.T)[:, :3]
    
    lines = [
        [0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 4], [4, 5], [5, 6]
    ]
    
    # Generate points along each line and collect them
    grasp_points = []
    for line in lines:
        start, end = pts[line[0]], pts[line[1]]
        line_points = generate_points_along_line(start, end, num_points=num_points_per_line)
        grasp_points.append(line_points)
    
    grasp_points = np.vstack(grasp_points)
    all_points.append(grasp_points)

    all_points = np.vstack(all_points)




    # if colors is not None:
    #     colors = np.repeat(colors, num_points_per_line * 8, axis=0)
    # else:
    #     colors = np.repeat(np.array([color]), all_points.shape[0], axis=0)
    
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(all_points)

    #Colorize the point cloud uniformly
    pcd.colors = o3d.utility.Vector3dVector([color] * len(all_points))


    return pcd


def get_grasp_color(i):
    """
    Assigns a different color to each point cloud in the grasp_pointclouds list.
    
    Colors used: white, black, green, blue, red, purple, orange, dark green, dark blue, brown, pink
    
    :param grasp_pointclouds: List of open3d.geometry.PointCloud objects
    :return: List of colored open3d.geometry.PointCloud objects
    """
    colors = [
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [1.0, 0.0, 0.0],  # red
        [0.0, 0.0, 0.0],  # black
        [1.0, 1.0, 1.0],  # white
        [0.5, 0.1, 0.5],  # purple
        [1.0, 0.65, 0.0], # orange
        [0.0, 0.0, 0.5],  # dark blue
        [1.0, 0.75, 0.8], # pink
        [0.0, 0.5, 0.0],  # dark green
        [0.6, 0.3, 0.1],  # brown
    ]
    
    color = colors[i % len(colors)]
  
    return color


    