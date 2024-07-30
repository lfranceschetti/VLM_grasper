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

def print_camera_parameters(vis):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    print("INTRINSIC MATRIX:\n", params.intrinsic.intrinsic_matrix)
    print("EXTRINSIC MATRIX:\n", params.extrinsic)
    return False

def set_extrinsic_and_take_image(vis, extrinsic, file_name):
    ctr = vis.get_view_control()
    camera_parameters = o3d.camera.PinholeCameraParameters()

    camera_parameters.extrinsic = np.array(extrinsic)

    camera_parameters.intrinsic = ctr.convert_to_pinhole_camera_parameters().intrinsic

    ctr.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)

    vis.update_renderer()

    image = vis.capture_screen_float_buffer(do_render=True)

    # Convert to numpy array and save the image
    image = (255 * np.asarray(image)).astype(np.uint8)

    # Save the image
    plt.imsave(file_name, image)





def visualize_grasps(full_pc, pred_grasps_cam, scores, pc_colors=None, gripper_openings=None, gripper_width=0.08, extrinsics=None, intrinsics=None):
    """
    Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions. 
    Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

    Arguments:
        full_pc {np.ndarray} -- Nx3 point cloud of the scene
        pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
        scores {dict[int:np.ndarray]} -- Confidence scores for grasps
        extrinsics {np.ndarray} -- Camera extrinsics in [x, y, z, qx, qy, qz, qw] format (default: {None})
        intrinsics {np.ndarray} -- Camera intrinsics 3x3 matrix (default: {None})


    Keyword Arguments:
        pc_colors {np.ndarray} -- Nx3 point cloud colors (default: {None})
        gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
        gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.08})
    """

    print('Visualizing...takes time')
  
    geometries = get_geometries(full_pc, pred_grasps_cam, scores, pc_colors, gripper_openings, gripper_width)

    # Visualize all geometries
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=True)  

    # Add geometries to the visualizer
    for geometry in geometries:
        vis.add_geometry(geometry)


    vis.update_renderer()

    vis.register_key_callback(ord("P"), print_camera_parameters)



    vis.run()
    vis.destroy_window()
   



def get_geometries(full_pc, pred_grasps_cam, scores, pc_colors=None, gripper_openings=None, gripper_width=0.08):
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('gist_rainbow')

    # Create point cloud
    pcd = draw_pc_with_colors(full_pc, pc_colors)
    geometries = [pcd]

    # Generate colors for the grasps
    colors = [cm(1. * i/len(pred_grasps_cam))[:3] for i in range(len(pred_grasps_cam))]
    colors2 = {k: cm2(0.5 * np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}
    
    # Add grasps to the geometries
    for i, k in enumerate(pred_grasps_cam):
        if np.any(pred_grasps_cam[k]):
            gripper_openings_k = np.ones(len(pred_grasps_cam[k])) * gripper_width if gripper_openings is None else gripper_openings[k]
            if len(pred_grasps_cam) > 1:
                geometries.append(draw_grasps(pred_grasps_cam[k], np.eye(4), gripper_openings_k, color=colors[i]))
                geometries.append(draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), [gripper_openings_k[np.argmax(scores[k])]], color=colors2[k], tube_radius=0.0025))
            else:
                colors3 = [cm2(0.5 * score)[:3] for score in scores[k]]
                geometries.append(draw_grasps(pred_grasps_cam[k], np.eye(4), gripper_openings_k, colors=colors3))
    
    return geometries

def draw_pc_with_colors(pc, pc_colors=None, single_color=(0.3,0.3,0.3), mode='2dsquare', scale_factor=0.0018):
    """
    Draws colored point clouds with open3d

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        pc_colors {np.ndarray} -- Nx3 point cloud colors

    Keyword Arguments:
        single_color {tuple} -- single color for point cloud (default: {(0.3,0.3,0.3)})
        mode {str} -- primitive type to plot (default: {'point'})
        scale_factor {float} -- Scale of primitives. Does not work for points. (default: {0.002})

    """

    if pc_colors is None:
        pc_colors = np.ones_like(pc) * single_color
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    
    return pcd

def generate_points_along_line(start, end, num_points=100):
    """
    Generates a series of points along a line between start and end points.
    
    Arguments:
        start {np.ndarray} -- 3D start point of the line
        end {np.ndarray} -- 3D end point of the line
        num_points {int} -- Number of points to generate along the line
    
    Returns:
        np.ndarray -- Array of points along the line
    """
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
    
    grasp_pcd = o3d.geometry.PointCloud()
    grasp_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    grasp_pcd.colors = o3d.utility.Vector3dVector([color] * len(grasp_pcd.points))

    return grasp_pcd
    

def main():
    # Get the package path using rospkg
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vlm_grasper')

    # Load data from files
    points = np.load(os.path.join(package_path, "src", "vlm_grasper", "data", "points.npy"), allow_pickle=True)
    transformation_matrices = np.load(os.path.join(package_path, "src", "vlm_grasper", "data", "transformation_matrices.npy"), allow_pickle=True)
    scores = np.load(os.path.join(package_path, "src", "vlm_grasper", "data", "scores.npy"), allow_pickle=True)
    intrinsics = np.load(os.path.join(package_path, "src", "vlm_grasper", "data", "intrinsics.npy"), allow_pickle=True)
    extrinsics = np.load(os.path.join(package_path, "src", "vlm_grasper", "data", "extrinsics.npy"), allow_pickle=True)


    scores = {0: scores}
    transformation_matrices = {0: transformation_matrices}


    # Call the visualize_grasps function
    visualize_grasps(points,transformation_matrices, scores, extrinsics=extrinsics, intrinsics=intrinsics)

if __name__ == "__main__":
    main()