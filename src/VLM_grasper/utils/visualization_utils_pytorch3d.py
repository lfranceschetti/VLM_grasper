#Taken from
# https://github.com/NVlabs/contact_graspnet/tree/main


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PointsRasterizationSettings, 
    PointsRenderer, 
    AlphaCompositor, 
    PointsRasterizer,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    HardPhongShader,
    Textures
)
import pytorch3d
import open3d as o3d
import VLM_grasper.utils.mesh_utils as mesh_utils



print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch3D version:", pytorch3d.__version__)



def create_cylinder_between_points(p1, p2, radius=0.005, resolution=10):
    p1, p2 = np.array(p1), np.array(p2)
    direction = p2 - p1
    height = np.linalg.norm(direction)
    direction = direction / height
    t = np.linspace(0, height, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    t, theta = np.meshgrid(t, theta)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = t
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    
    vertices = np.stack([x, y, z], axis=1)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            faces.append([i*resolution+j, (i+1)*resolution+j, i*resolution+j+1])
            faces.append([(i+1)*resolution+j, (i+1)*resolution+j+1, i*resolution+j+1])
    
    faces = np.array(faces)
    
    # Apply transformation to align with direction
    z_axis = np.array([0, 0, 1])
    rotation_matrix = np.linalg.lstsq([z_axis], [direction], rcond=None)[0]
    vertices = vertices.dot(rotation_matrix.T)
    
    vertices += p1
    return vertices, faces

                
def visualize_grasps(full_pc, pred_grasps_cam, scores, plot_opencv_cam=False, pc_colors=None, gripper_openings=None, gripper_width=0.08):
    print('Visualizing...takes time')
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('gist_rainbow')
   
    colors = [cm(1. * i / len(pred_grasps_cam))[:3] for i in range(len(pred_grasps_cam))]
    colors2 = {k: cm2(0.5 * np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}
    
    grasp_meshes = []
    for i, k in enumerate(pred_grasps_cam):
        if np.any(pred_grasps_cam[k]):
            gripper_openings_k = np.ones(len(pred_grasps_cam[k])) * gripper_width if gripper_openings is None else gripper_openings[k]
            grasp_mesh = draw_grasps(pred_grasps_cam[k], np.eye(4), gripper_openings_k, color=colors[i])
            if grasp_mesh:
                grasp_meshes.append(grasp_mesh)
                grasp_mesh_confident = draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), [gripper_openings_k[np.argmax(scores[k])]], color=colors2[k], tube_radius=0.0025)
                if grasp_mesh_confident:
                    grasp_meshes.append(grasp_mesh_confident)
    
    if grasp_meshes:
        grasp_meshes = Meshes(
            verts=[m.verts_list()[0] for m in grasp_meshes],
            faces=[m.faces_list()[0] for m in grasp_meshes],
            textures=[m.textures for m in grasp_meshes]
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_pc)
    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    else:
        pcd.paint_uniform_color([0.3, 0.3, 0.3])

    camera_pose = np.eye(4)  # Set your desired camera pose here
    render_scene(pcd, grasp_meshes, camera_pose)  
 



def draw_grasps(grasps, cam_pose, gripper_openings, color=(0, 1., 0), colors=None, show_gripper_mesh=False, tube_radius=0.0008):
    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    all_verts = []
    all_faces = []
    for i, (g, g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * g_opening / 2

        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        
        color = color if colors is None else colors[i]
        
        for j in range(len(pts) - 1):
            verts, faces = create_cylinder_between_points(pts[j], pts[j + 1], radius=tube_radius)
            all_verts.append(verts)
            all_faces.append(faces + len(np.vstack(all_verts)))
    
    if len(all_verts) > 0:
        all_verts = np.vstack(all_verts)
        all_faces = np.vstack(all_faces)
        
        textures = Textures(verts_rgb=torch.ones_like(torch.tensor(all_verts, dtype=torch.float32))[None] * torch.tensor(color, dtype=torch.float32))
        grasp_mesh = Meshes(verts=[torch.tensor(all_verts, dtype=torch.float32)], faces=[torch.tensor(all_faces, dtype=torch.int64)], textures=textures)
        
        return grasp_mesh
    else:
        return None

def render_scene(point_cloud, grasp_meshes, camera_pose):
    device = torch.device("cuda:0")
    
    # Set up the camera
    cameras = FoVPerspectiveCameras(R=camera_pose[:3, :3].unsqueeze(0), T=camera_pose[:3, 3].unsqueeze(0), device=device)
    
    # Set up the point cloud renderer
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.005,
        points_per_pixel=10
    )
    
    pointcloud = Pointclouds(points=[torch.tensor(point_cloud.points, dtype=torch.float32).to(device)],
                             features=[torch.tensor(point_cloud.colors, dtype=torch.float32).to(device)])
    
    pointcloud_renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    
    # Render the point cloud
    images = pointcloud_renderer(pointcloud)
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.show()
    
    # Set up the mesh renderer for grasp visualization
    if grasp_meshes is not None:
        raster_settings_mesh = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        
        mesh_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_mesh),
            shader=HardPhongShader(device=device, cameras=cameras)
        )
        
        # Render the grasp meshes
        images_mesh = mesh_renderer(grasp_meshes.to(device))
        plt.imshow(images_mesh[0, ..., :3].cpu().numpy())
        plt.show()




