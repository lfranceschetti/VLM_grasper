#!/usr/bin/env python
import numpy as np
import open3d as o3d
import torch
from vlm_grasper.models.vn_edge_grasper import EdgeGrasper as VNGrasper
from torch_geometric.data import Data
from torch_geometric.nn import radius
import torch_geometric
from vlm_grasper.simulator.utility import FarthestSamplerTorch, get_gripper_points_mask, orthognal_grasps, FarthestSampler
import torch.nn.functional as F
import rospy
import rospkg
import os
import sys

SAMPLE_NUMBER = 64



def perform_edge_grasp(pc, surface_normal= None, add_noise=False, plotting=False):

    
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vlm_grasper')   
    root_dir = os.path.join(package_path, 'src/vlm_grasper/vn_edge_pretrained_para') 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNGrasper(device=1, root_dir=root_dir, load=105)


    if plotting:
        o3d.visualization.draw_geometries([pc])

    if pc.is_empty():
        rospy.logwarn("Point cloud is empty")
        return
    
    if add_noise:
        vertices = np.asarray(pc.points)
        # add gaussian noise 95% confident interval (-1.96,1.96)
        vertices = vertices + np.random.normal(loc=0.0, scale=0.0008, size=(len(vertices), 3))
        pc.points = o3d.utility.Vector3dVector(vertices)


    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))

    if pc.has_normals():
        pc.orient_normals_consistent_tangent_plane(30)
    else:
        # Handle the case where normals are not computed
        rospy.logwarn("Normals not computed, computing normals")

    pc = pc.voxel_down_sample(voxel_size=0.0045)
        

    pos = np.asarray(pc.points)
    normals = np.asarray(pc.normals)

    pos = torch.from_numpy(pos).to(torch.float32).to(device)
    # print('min z, max z', pos[:,-1].min(), pos[:,-1].max())
    normals = torch.from_numpy(normals).to(torch.float32).to(device)
    sample_number = SAMPLE_NUMBER

    fps_sample = FarthestSamplerTorch()
    _, sample = fps_sample(pos,sample_number)
    sample = torch.as_tensor(sample).to(torch.long).reshape(-1).to(device)
    sample = torch.unique(sample,sorted=True)
    #print(sample)
    #sample = np.random.choice(len(pos), sample_number, replace=False)
    #sample = torch.from_numpy(sample).to(torch.long)
    sample_pos = pos[sample, :]
    radius_p_batch_index = radius(pos, sample_pos, r=0.05, max_num_neighbors=1024)
    radius_p_index = radius_p_batch_index[1, :].to(device)
    radius_p_batch = radius_p_batch_index[0, :].to(device)
    sample_pos = torch.cat(
        [sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
        dim=0)
    sample_copy = sample.clone().unsqueeze(dim=-1)
    sample_index = torch.cat(
        [sample_copy[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))], dim=0)
    edges = torch.cat((sample_index, radius_p_index.unsqueeze(dim=-1)), dim=1)
    #all_edge_index = numpy.arange(0, len(edges))
    all_edge_index = torch.arange(0,len(edges)).to(device)
    des_pos = pos[radius_p_index, :]
    des_normals = normals[radius_p_index, :]
    relative_pos = des_pos - sample_pos


    relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)


    # only record approach vectors with a angle mask
    x_axis = torch.cross(des_normals, relative_pos_normalized)
    x_axis = F.normalize(x_axis, p=2, dim=1)
    valid_edge_approach = torch.cross(x_axis, des_normals)
    valid_edge_approach = F.normalize(valid_edge_approach, p=2, dim=1)
    valid_edge_approach = -valid_edge_approach
    up_dot_mask = torch.einsum('ik,k->i', valid_edge_approach, torch.tensor([0., 0., 1.]).to(device))
    relative_norm = torch.linalg.norm(relative_pos, dim=-1)
    depth_proj = -torch.sum(relative_pos * valid_edge_approach, dim=-1)

    # Define the condition masks
    angle_condition = up_dot_mask > -0.1
    distance_condition = torch.logical_and(relative_norm > 0.003, relative_norm < 0.038)
    depth_condition = torch.logical_and(depth_proj > -0.000, depth_proj < 0.04)

    # Combine conditions into the geometry mask
    geometry_mask = torch.logical_and(angle_condition, distance_condition)
    geometry_mask = torch.logical_and(geometry_mask, depth_condition)

    # Log the results of each condition

    rospy.set_param("n_grasps_beginning", len(geometry_mask))

    rospy.set_param("n_graps_geometry_mask", geometry_mask.sum().item())



    # draw_grasps2(geometry_mask, depth_proj, valid_edge_approach, des_normals, sample_pos, pos, sample, des=None, scores=None)
    pose_candidates = orthognal_grasps(geometry_mask, depth_proj, valid_edge_approach, des_normals,
                                        sample_pos)
    

    #Create a mask to check if the grasp is coming from below for this you should check the angle between the approach vector and the surface normal
    #If the angle is greater than 90 degrees, then the grasp is coming from below
    #If the angle is less than 90 degrees, then the grasp is coming from above
    if surface_normal is not None:
        surface_normal = torch.from_numpy(surface_normal).to(torch.float32).to(device)
    
    # Count how many pose candidates pass the overall geometry filter
    pre_filter_count = geometry_mask.sum().item()

    rospy.set_param("n_grasps_surface_normal", pre_filter_count)

    # Apply the additional filter for checking if grasps are coming from below
    angle = torch.einsum('ik,k->i', valid_edge_approach, surface_normal)
    mask = angle > 0

    # Count how many pose candidates pass after applying the additional filter
    post_filter_count = mask.sum().item()

    # Apply the mask to the pose candidates and update geometry_mask
    geometry_mask = torch.logical_and(geometry_mask, mask)

    post_filter_count = geometry_mask.sum().item()

    rospy.loginfo(f"Pose candidates passing additional filter: {post_filter_count}")

    edge_sample_index = all_edge_index[geometry_mask]
    # print('no collision with table candidates', len(edge_sample_index))
    if len(edge_sample_index) > 0:
        if len(edge_sample_index) > 1500:
            edge_sample_index = edge_sample_index[torch.randperm(len(edge_sample_index))[:1500]]
        edge_sample_index, _ = torch.sort(edge_sample_index)
        # print('candidate numbers', len(edge_sample_index))
        data = Data(pos=pos, normals=normals, sample=sample, radius_p_index=radius_p_index,
                    ball_batch=radius_p_batch,
                    ball_edges=edges, approaches=valid_edge_approach[edge_sample_index, :],
                    reindexes=edge_sample_index,
                    relative_pos=relative_pos[edge_sample_index, :],
                    depth_proj=depth_proj[edge_sample_index])

        data = data.to(device)
        grasps = model.model.act(data)

        return grasps
    else:
        rospy.logwarn("The edge sample index is empty")
        return None

  