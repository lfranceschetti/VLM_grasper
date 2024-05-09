#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import torch
from VLM_grasper.models.vn_edge_grasper import EdgeGrasper as VNGrasper
from torch_geometric.data import Data
from torch_geometric.nn import radius
import torch_geometric
from VLM_grasper.simulator.utility import FarthestSamplerTorch, get_gripper_points_mask, orthognal_grasps, FarthestSampler
import torch.nn.functional as F
import rospy
import os

SAMPLE_NUMBER = 64



def perform_edge_grasp(pc, add_noise=False, plotting=False):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNGrasper(device=1, root_dir='/home/lucfra/semester-project/catkin_ws/src/VLM_grasper/src/VLM_grasper/vn_edge_pretrained_para', load=105)


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


    # COMMENTED OUT FOR NOW
    # pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    # pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
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
    geometry_mask = torch.logical_and(up_dot_mask > -0.1, relative_norm > 0.003)
    geometry_mask = torch.logical_and(relative_norm<0.038,geometry_mask)
    depth_proj_mask = torch.logical_and(depth_proj > -0.000, depth_proj < 0.04)
    geometry_mask = torch.logical_and(geometry_mask, depth_proj_mask)

    # draw_grasps2(geometry_mask, depth_proj, valid_edge_approach, des_normals, sample_pos, pos, sample, des=None, scores=None)
    pose_candidates = orthognal_grasps(geometry_mask, depth_proj, valid_edge_approach, des_normals,
                                        sample_pos)
    table_grasp_mask = get_gripper_points_mask(pose_candidates,threshold=0.054)

    # print('no collision with table candidates all', table_grasp_mask.sum())
    geometry_mask[geometry_mask == True] = table_grasp_mask
    # wether fps
    # geometry_mask = torch.logical_and(geometry_mask,geometry_mask2)
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

        print('grasps', grasps)

        return grasps
    else:
        rospy.logwarn("The edge sample index is empty")
        return None

  