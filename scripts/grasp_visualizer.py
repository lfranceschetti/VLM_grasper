#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from VLM_grasper.msg import PointCloudWithGrasps

def grasp_callback(pc_with_grasps, args):
    marker_pub = args[0]

    rospy.loginfo(f"Received point cloud data from frame: {pc_with_grasps.point_cloud.header.frame_id}")
    rospy.loginfo(f"Received grasps: {len(pc_with_grasps.grasps)}")
    # Assume point_cloud is a PointCloud2 message that needs to be processed
    # gen = pc2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z"))
    # points = np.array(list(gen))
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(points)

    # # Visualization logic here, potentially creating markers for each pose in pose_array
    # marker_array = MarkerArray()
    # for i, pose in enumerate(pose_array.poses):
    #     marker = Marker()
    #     marker.header.frame_id = pose_array.header.frame_id
    #     marker.type = marker.SPHERE
    #     marker.action = marker.ADD
    #     marker.pose = pose
    #     marker.scale.x = marker.scale.y = marker.scale.z = 0.02  # size of the marker
    #     marker.color.a = 1.0  # Don't forget to set the alpha!
    #     marker.color.r = 1.0
    #     marker.color.g = 0.0
    #     marker.color.b = 0.0
    #     marker_array.markers.append(marker)

    # # Publish your markers to a topic
    # marker_pub.publish(marker_array)

def main():
    rospy.init_node('grasp_visualizer')
    marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
    pose_sub = rospy.Subscriber('/point_cloud_with_grasps', PointCloudWithGrasps, grasp_callback, callback_args=(marker_pub,))
    rospy.spin()

if __name__ == '__main__':
    main()
