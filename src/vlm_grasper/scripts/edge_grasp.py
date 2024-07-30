#!/home/lucfra/miniconda3/envs/ros_env/bin/python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import message_filters

import numpy as np
import open3d as o3d
from vlm_grasper.utils.perform_edge_grasp import perform_edge_grasp


from vlm_grasper.msg import PointCloudWithGrasps
from vlm_grasper.msg import Grasp
from geometry_msgs.msg import Vector3Stamped, Vector3



def create_grasp_msg(score, sample_pos, depth_projection, approach, normal):
    grasp = Grasp()
    grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z = sample_pos
    # Assuming the orientation needs to be calculated or is provided
    grasp.score = score
    grasp.depth_projection = depth_projection
    grasp.approach = Vector3(*approach)
    grasp.normal = Vector3(*normal)
    return grasp



def callback(pc_msg, surface_normal_msg, args):

    pub = args[0]

    surface_normal = np.array([surface_normal_msg.vector.x, surface_normal_msg.vector.y, surface_normal_msg.vector.z])

    # Convert ROS PointCloud2 to Open3D PointCloud
    gen = pc2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z"))
    points = np.array(list(gen))
    pc = o3d.geometry.PointCloud()
    
    pc.points = o3d.utility.Vector3dVector(points)

    # Process the point cloud and predict grasps
    grasp_data = perform_edge_grasp(pc, add_noise=True, plotting=False)

    if grasp_data is None:
        rospy.logwarn("No grasps found")
        return
    else:
        rospy.loginfo(f"Found {len(grasp_data[0])} grasps")

    score, depth_projection, approaches, sample_pos, des_normals = grasp_data

    approaches_numpy = approaches.cpu().numpy()

    possible_indices = [i for i, approach in enumerate(approaches.cpu().numpy()) if np.dot(approach, surface_normal) > 0]

    # FOR NOW TAKE THE 10 WITH THE HIGHEST SCORE

    # Ensure each step is explicitly copied if necessary
    score = score.cpu().detach().numpy()
    #Take the indices of the 10 highest scores
    sorted_indices = np.argsort(score)[::-1]
  
    best_indices = [i for i in sorted_indices if i in possible_indices][:5]
    score = score[best_indices].copy()
    sample_pos = sample_pos.cpu().numpy()[best_indices].copy()  # Force a copy here
    des_normals = des_normals.cpu().numpy()[best_indices].copy()
    depth_projection = depth_projection.cpu().numpy()[best_indices].copy()
    approaches = approaches.cpu().numpy()[best_indices].copy()

    rospy.loginfo(f"Publishing {len(score)} grasps")

   
    grasps_msg = [create_grasp_msg(pos, score, dp, app, norm)
                  for pos, score, dp, app, norm in zip(score, sample_pos, depth_projection, approaches, des_normals)]



    combined_msg = PointCloudWithGrasps()
    combined_msg.header.stamp = pc_msg.header.stamp  # Preserve the original timestamp
    combined_msg.header.frame_id = pc_msg.header.frame_id
    combined_msg.point_cloud = pc_msg
    combined_msg.grasps = grasps_msg

    pub.publish(combined_msg)

def main():
    rospy.init_node('edge_grasp')
    rospy.loginfo("Edge grasp node started")
    pub = rospy.Publisher('/point_cloud_with_grasps', PointCloudWithGrasps, queue_size=10)
    pc_sub = message_filters.Subscriber("/point_cloud/processed", PointCloud2)
    surface_normal_msg = message_filters.Subscriber("/surface_normal", Vector3Stamped)

    ts = message_filters.ApproximateTimeSynchronizer([pc_sub, surface_normal_msg], 10, 0.1)
    ts.registerCallback(callback, (pub,))

    rospy.spin()

if __name__ == '__main__':
    main()


