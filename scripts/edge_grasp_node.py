#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from VLM_grasper.utils.perform_edge_grasp import perform_edge_grasp


from VLM_grasper.msg import PointCloudWithGrasps
from VLM_grasper.msg import Grasp
from geometry_msgs.msg import Vector3



def create_grasp_msg(score, sample_pos, depth_projection, approach, normal):
    grasp = Grasp()
    grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z = sample_pos
    # Assuming the orientation needs to be calculated or is provided
    grasp.score = score
    grasp.depth_projection = depth_projection
    grasp.approach = Vector3(*approach)
    grasp.normal = Vector3(*normal)
    return grasp



def callback(data, args):

    pub = args[0]

    rospy.loginfo(f"Received point cloud data from frame: {data.header.frame_id}")
    # Convert ROS PointCloud2 to Open3D PointCloud
    gen = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
    points = np.array(list(gen))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # Process the point cloud and predict grasps
    grasp_data = perform_edge_grasp(pc, add_noise=True, plotting=False)

    if grasp_data is None:
        rospy.logwarn("No grasps found")
        return

    score, depth_projection, approaches, sample_pos, des_normals = grasp_data


    # FOR NOW TAKE THE 10 WITH THE HIGHEST SCORE

    # Ensure each step is explicitly copied if necessary
    score = score.cpu().detach().numpy()
    sorted_indices = np.argsort(-score)  # More straightforward, avoids reversing
    best_indices = sorted_indices[:10]
    score = score[best_indices].copy()
    sample_pos = sample_pos.cpu().numpy()[best_indices].copy()  # Force a copy here
    des_normals = des_normals.cpu().numpy()[best_indices].copy()
    depth_projection = depth_projection.cpu().numpy()[best_indices].copy()
    approaches = approaches.cpu().numpy()[best_indices].copy()

   
    grasps_msg = [create_grasp_msg(pos, score, dp, app, norm)
                  for pos, score, dp, app, norm in zip(score, sample_pos, depth_projection, approaches, des_normals)]


    combined_msg = PointCloudWithGrasps()
    combined_msg.header.stamp = rospy.Time.now()
    combined_msg.header.frame_id = data.header.frame_id
    combined_msg.point_cloud = data
    combined_msg.grasps = grasps_msg

    pub.publish(combined_msg)

def main():
    rospy.init_node('edge_grasp_node')
    rospy.loginfo("Edge grasp node started")
    pub = rospy.Publisher('/point_cloud_with_grasps', PointCloudWithGrasps, queue_size=10)
    rospy.Subscriber("/input_point_cloud", PointCloud2, callback, callback_args=(pub,))
    rospy.spin()

if __name__ == '__main__':
    main()


