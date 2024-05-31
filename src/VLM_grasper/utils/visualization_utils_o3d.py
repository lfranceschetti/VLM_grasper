#Taken from
# https://github.com/NVlabs/contact_graspnet/tree/main


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d as o3d
import rospkg
import os


import VLM_grasper.utils.mesh_utils as mesh_utils



                
def show_image(rgb, segmap):
    """
    Overlay rgb image with segmentation and imshow segment

    Arguments:
        rgb {np.ndarray} -- color image
        segmap {np.ndarray} -- integer segmap of same size as rgb
    """
    plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    plt.ion()
    plt.show()
    
    if rgb is not None:
        plt.imshow(rgb)
    if segmap is not None:
        cmap = plt.get_cmap('rainbow')
        cmap.set_under(alpha=0.0)   
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)
    plt.draw()
    plt.pause(0.001)

def visualize_grasps(full_pc, pred_grasps_cam, scores, plot_opencv_cam=False, pc_colors=None, gripper_openings=None, gripper_width=0.08):
    """
    Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions. 
    Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

    Arguments:
        full_pc {np.ndarray} -- Nx3 point cloud of the scene
        pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
        scores {dict[int:np.ndarray]} -- Confidence scores for grasps

    Keyword Arguments:
        plot_opencv_cam {bool} -- plot camera coordinate frame (default: {False})
        pc_colors {np.ndarray} -- Nx3 point cloud colors (default: {None})
        gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
        gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.08})
    """

    print('Visualizing...takes time')
  
    geometries = get_geometries(full_pc, pred_grasps_cam, scores, pc_colors, gripper_openings, gripper_width)
    # Visualize all geometries
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Create a window, but do not display it
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Set camera parameters manually
    ctr = vis.get_view_control()
    pinhole_camera_parameters = o3d.camera.PinholeCameraParameters()

    # Define camera intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=640, height=640, fx=525.0, fy=525.0, cx=320.0, cy=240.0)
    pinhole_camera_parameters.intrinsic = intrinsic

    # Define camera extrinsic parameters
    extrinsic = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0, 1.0]])
    pinhole_camera_parameters.extrinsic = extrinsic

    ctr.convert_from_pinhole_camera_parameters(pinhole_camera_parameters)

    vis.poll_events()
    vis.update_renderer()
    
    # Capture the image
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    # Convert to numpy array and save the image
    image = (255 * np.asarray(image)).astype(np.uint8)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    plt.imsave('rendered_image.png', image)

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
                geometries += draw_grasps(pred_grasps_cam[k], np.eye(4), gripper_openings_k, color=colors[i])
                geometries += draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), [gripper_openings_k[np.argmax(scores[k])]], color=colors2[k], tube_radius=0.0025)
            else:
                colors3 = [cm2(0.5 * score)[:3] for score in scores[k]]
                geometries += draw_grasps(pred_grasps_cam[k], np.eye(4), gripper_openings_k, colors=colors3)
    
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


def plot_mesh(mesh, cam_pose, grasp_pose):
        mesh.transform(grasp_pose)
        mesh.transform(cam_pose)
        return mesh

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
        show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
    """

    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    if show_gripper_mesh and len(grasps) > 0:
        plot_mesh(gripper.hand, cam_pose, grasps[0])
        
    geometries = []
    
    if show_gripper_mesh and len(grasps) > 0:
        gripper_mesh = o3d.geometry.TriangleMesh.create_box()  # Placeholder for actual gripper mesh
        gripper_mesh = plot_mesh(gripper_mesh, cam_pose, grasps[0])
        geometries.append(gripper_mesh)
    
    for i, (g, g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
        
        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))), axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:, :3]
        
        color = color if colors is None else colors[i]
        color = tuple(color)
        
        lines = [
            [0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 4], [4, 5], [5, 6]
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

        geometries.append(line_set)
    
    return geometries
    

def main():
    # Get the package path using rospkg
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('VLM_grasper')

    # Load data from files
    points = np.load(os.path.join(package_path, "src", "VLM_grasper", "data", "points.npy"), allow_pickle=True)
    transformation_matrices = np.load(os.path.join(package_path, "src", "VLM_grasper", "data", "transformation_matrices.npy"), allow_pickle=True)
    scores = np.load(os.path.join(package_path, "src", "VLM_grasper", "data", "scores.npy"), allow_pickle=True)


    scores = {0: scores}
    transformation_matrices = {0: transformation_matrices}




    # Call the visualize_grasps function
    visualize_grasps(points,transformation_matrices, scores)

if __name__ == "__main__":
    main()