#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from vlm_grasper.utils.perform_edge_grasp import perform_edge_grasp
from vlm_grasper.utils.render_helpers import create_transformation_matrix
from vlm_grasper.msg import PointCloudWithGrasps, Grasp
from geometry_msgs.msg import Vector3Stamped, Vector3
import time

class EdgeGraspNode:
    def __init__(self):
        rospy.init_node('edge_grasp')
        rospy.loginfo("Edge grasp node started")

        self.pc_msg = None
        self.surface_normal_msg = None

        self.publisher_all = rospy.Publisher('/point_cloud_with_grasps/all', PointCloudWithGrasps, queue_size=30)
        self.publisher_score_filter = rospy.Publisher('/point_cloud_with_grasps/score_filter', PointCloudWithGrasps, queue_size=30)

        rospy.Subscriber("/point_cloud/processed", PointCloud2, self.pc_callback)
        rospy.Subscriber("/surface_normal", Vector3Stamped, self.surface_normal_callback)

    def pc_callback_sim(self, msg):
        self.pc_msg = msg
        self.try_process_grasp()

    def pc_callback(self, msg):
        self.pc_msg = msg
        self.try_process_grasp()

    def surface_normal_callback(self, msg):
        self.surface_normal_msg = msg
        self.try_process_grasp()

    def try_process_grasp(self):
        if self.pc_msg and (self.surface_normal_msg or rospy.get_param("use_edge_grasp_sim", False)):
            rospy.loginfo("Edge Grasp: Trying to process grasp")
            rospy.set_param('point_cloud_transformed', True)
            if not rospy.get_param('edge_grasp_finished', True):
                rospy.loginfo("Edge Grasp: Processing grasp")
                self.process_grasp(self.pc_msg, self.surface_normal_msg)
                rospy.set_param("edge_grasp_finished", True)


    def process_grasp(self, pc_msg, surface_normal_msg):
        surface_normal = np.array([surface_normal_msg.vector.x, surface_normal_msg.vector.y, surface_normal_msg.vector.z]) if surface_normal_msg else None

        # Convert ROS PointCloud2 to Open3D PointCloud
        gen = pc2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z"))
        points = np.array(list(gen))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        # Process the point cloud and predict grasps
        grasp_data = perform_edge_grasp(pc, surface_normal, add_noise=False, plotting=False)

        if grasp_data is None:
            rospy.logwarn("No grasps found")
            return
  

        score, depth_projection, approaches, sample_pos, des_normals = grasp_data

        # Ensure each step is explicitly copied if necessary
        score = score.cpu().detach().numpy().copy()
        sample_pos = sample_pos.cpu().numpy().copy()
        des_normals = des_normals.cpu().numpy().copy()
        depth_projection = depth_projection.cpu().numpy().copy()
        approaches = approaches.cpu().numpy().copy()

        # Sort according to score
        idx = np.argsort(-score)
        score = score[idx]
        sample_pos = sample_pos[idx]
        des_normals = des_normals[idx]
        depth_projection = depth_projection[idx]
        approaches = approaches[idx]

        grasps_msg = [self.create_grasp_msg(pos, score, dp, app, norm)
                      for pos, score, dp, app, norm in zip(score, sample_pos, depth_projection, approaches, des_normals)]

        combined_msg = PointCloudWithGrasps()
        combined_msg.header.stamp = pc_msg.header.stamp  # Preserve the original timestamp
        combined_msg.header.frame_id = pc_msg.header.frame_id
        combined_msg.point_cloud = pc_msg
        combined_msg.grasps = grasps_msg
        self.publisher_all.publish(combined_msg)

        now = time.time() 
        rospy.set_param('t_edge_grasp_completed', now)

        # Take the 5 random grasps from the ones with scores above 0.9 and 5 grasps that are below 0.2
        grasps_above_09 = [grasp for grasp in grasps_msg if grasp.score > 0.9]
        grasps_below_02 = [grasp for grasp in grasps_msg if grasp.score < 0.2]
        for grasp in grasps_below_02:
            grasp.score = -1.0
        #Take 3 at random from those that are above and below
      
        
        combined_grasps = list(grasps_above_09) + list(grasps_below_02)


        combined_msg.grasps = combined_grasps
        self.publisher_score_filter.publish(combined_msg)
        self.reset()

    @staticmethod
    def create_grasp_msg(score, sample_pos, depth_projection, approach, normal):
        grasp = Grasp()
        grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z = sample_pos
        grasp.score = score
        grasp.depth_projection = depth_projection
        grasp.approach = Vector3(*approach)
        grasp.normal = Vector3(*normal)
        return grasp
    
    def reset(self):
        self.pc_msg = None
        self.surface_normal_msg = None


if __name__ == '__main__':
    EdgeGraspNode()
    rate = rospy.Rate(10)  # 10 Hz, for example
    
    while not rospy.is_shutdown():
        # Your main loop processing goes here
        
        rate.sleep()  # Maintain the loop at the specified rate
