#Taken from
# https://github.com/NVlabs/contact_graspnet/tree/main


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d as o3d
import rospkg
import os
from scipy.spatial.transform import Rotation as R
import vlm_grasper.utils.mesh_utils as mesh_utils




def get_grasp_pointclouds(pred_grasps_cam, scores, pc_colors=None, gripper_openings=None, gripper_width=0.08):
    
    grasp_pointclouds = []
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/len(pred_grasps_cam))[:3] for i in range(len(pred_grasps_cam))]
    colors2 = {k: cm2(0.5 * np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}


    
    # Add grasps to the geometries
    for i, k in enumerate(pred_grasps_cam):
        if np.any(pred_grasps_cam[k]):
            gripper_openings_k = np.ones(len(pred_grasps_cam[k])) * gripper_width if gripper_openings is None else gripper_openings[k]
            if len(pred_grasps_cam) > 1:
                grasp_pointclouds.append(draw_grasps(pred_grasps_cam[k], np.eye(4), gripper_openings_k, color=colors[i]))
                grasp_pointclouds.append(draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), [gripper_openings_k[np.argmax(scores[k])]], color=colors2[k], tube_radius=0.0025))
            else:
                colors3 = [cm2(0.5 * score)[:3] for score in scores[k]]
                grasp_pointclouds.append(draw_grasps(pred_grasps_cam[k], np.eye(4), gripper_openings_k, colors=colors3))

    print("Grasp point clouds: ", grasp_pointclouds)
    
    return grasp_pointclouds


def generate_points_along_line(start, end, num_points=100):

    return np.linspace(start, end, num_points)


def draw_grasps(grasps, cam_pose, gripper_openings, color=(0,1.,0), colors=None, show_gripper_mesh=False, tube_radius=0.0008):
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
    
    for i, (g, g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
        
        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
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
    if colors is not None:
        colors = np.repeat(colors, num_points_per_line * 8, axis=0)
    else:
        colors = np.repeat(np.array([color]), all_points.shape[0], axis=0)
    
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

    

    
    
def main():
    pass


if __name__ == "__main__":
    main()