#!/home/lucfra/miniconda3/envs/ros_env/bin/python
import rospy
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from VLM_grasper.msg import PointCloudWithGrasps
from VLM_grasper.utils.visualization_utils import visualize_grasps
import rospkg

import sys
import matplotlib.pyplot as plt
import os


LENGTH = 0.09

def create_transformation_matrix(position, approach, normal):
    # Convert attributes to arrays
    position = np.array([position.x, position.y, position.z])
    approach = np.array([approach.x, approach.y, approach.z])
    normal = np.array([normal.x, normal.y, normal.z])


    z_axis = approach
    #Make that the normal direction is flipped and shows in the other direction
    x_axis = -normal
    y_axis = np.cross(x_axis, z_axis)  # Compute the missing Y-axis

    # Normalize the axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)


    # Create rotation matrix (For some reason, the axes are flipped)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    rotation_matrix = np.dot(np.array([[-1,0,0], [0, -1, 0], [0, 0, -1]]), rotation_matrix)

    # Create translation vector 
    translation_vector = position
    # We need to move the gripper back by the length of the gripper
    translation_vector += LENGTH * approach
    

    # Combine into a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix    
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix

def grasp_callback(pc_with_grasps, args):
    marker_pub = args

    rospy.loginfo(f"Received grasps: {len(pc_with_grasps.grasps)}")


    #int:np.ndarray with transformation matrices
    transformation_matrices = []
    scores = []
    for grasp in pc_with_grasps.grasps:
        position = grasp.pose.position
        orientation = grasp.pose.orientation
        # If orientation is quaternion, we assume it's not used and instead use approach and normal
        approach = grasp.approach
        normal = grasp.normal

        score = grasp.score
        scores.append(score)

        matrix = create_transformation_matrix(position, approach, normal)
        transformation_matrices.append(matrix)


    scores = np.array(scores)
    transformation_matrices = np.array(transformation_matrices)


    scores = {0: scores}
    transformation_matrices = {0: transformation_matrices}

    print(transformation_matrices)

    # Generate point cloud data
    gen = pc2.read_points(pc_with_grasps.point_cloud, skip_nans=True, field_names=("x", "y", "z"))
    points = np.array(list(gen))

    # Now call visualize_grasps with the prepared data
    visualize_grasps(points, transformation_matrices, scores, plot_opencv_cam=True, pc_colors=None)




def main():
    rospy.init_node('grasp_visualizer')
    marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
    
    # Wait for a message to ensure there's data to visualize
    pc_with_grasps = rospy.wait_for_message('/point_cloud_with_grasps', PointCloudWithGrasps)
    
    # Process and visualize data immediately in the main thread
    grasp_callback(pc_with_grasps, marker_pub)
    rospy.spin()

if __name__ == '__main__':
    main()
