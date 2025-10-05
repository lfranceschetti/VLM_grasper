#!/usr/bin/env python

import sys
import math
import numpy as np
import rospy
import moveit_commander
import open3d as o3d

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from vlm_grasper.utils.grasp_simulator_helpers import transform_matrix, extract_pose_from_matrix, create_transformation_matrix

from vlm_grasper.msg import PointCloudWithGrasps, Grasp
from geometry_msgs.msg import Pose, Vector3
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
import tqdm
import time

    

def reach_pose(group, grasp, tolerance=0.01):

    transformation_matrix_camera_frame = create_transformation_matrix(grasp.pose.position, grasp.approach, grasp.normal)

    transformation_matrix_world_frame = transform_matrix(transformation_matrix_camera_frame, "camera_color_optical_frame", "world")

    pose_world_frame = extract_pose_from_matrix(transformation_matrix_world_frame)
    group.set_pose_target(pose_world_frame)


    group.set_goal_position_tolerance(tolerance)
    plan = group.plan()

    #Delete the target pose
    group.clear_pose_targets()


    position_found = plan[0]

    # Execute the plan
    if position_found:
        return True, "Works"
    else:
        return False, "No plan found"


def choice_of_grasps(grasps):
    """As a first grasp, choose the one with the highest score.
    Then, choose the one that varies the most from the first grasp."""

    choice_of_grasps = []
    if not grasps:
        return choice_of_grasps
    
    number_of_grasps = rospy.get_param("number_of_grasps", 5)

    if len(grasps) < number_of_grasps:
        return grasps

    best_grasp = max(grasps, key=lambda grasp: grasp.score)
    choice_of_grasps.append(best_grasp)

    grasps.remove(best_grasp)

    #Get the grasp that varies the most from all the other grasps
    for i in range(number_of_grasps-1):
        max_min_distance = 0
        best_index = 0
        for j, grasp in enumerate(grasps):
            min_distance = float('inf')
            for selected_grasp in choice_of_grasps:
                distance = np.linalg.norm(
                    np.array([grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z]) - 
                    np.array([selected_grasp.pose.position.x, selected_grasp.pose.position.y, selected_grasp.pose.position.z])
                )
                if distance < min_distance:
                    min_distance = distance
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_index = j

        choice_of_grasps.append(grasps[best_index])
        grasps.pop(best_index)


    return choice_of_grasps

            

def filter_by_score(grasps):
    """Filter grasps by score."""
    threshold = rospy.get_param("score_threshold", 0.9)

    rospy.loginfo(f"Number of Grasps above 0.9: {len([grasp for grasp in grasps if grasp.score > 0.9])}")

    return [grasp for grasp in grasps if grasp.score > threshold]



def callback(pc_with_grasps, args):

    pub = args[0]
    pub_moveit_filter = args[1]

       #Get point cloud as open3d point cloud
    pc_msg = pc_with_grasps.point_cloud
    gen = pc2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z"))
    points = np.array(list(gen))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    grasps = filter_by_score(pc_with_grasps.grasps)

    rospy.set_param("n_grasps_good_score", len(grasps))


    moveit_commander.roscpp_initialize(sys.argv)

    # Initialize MoveIt Commander
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("panda_arm")
    group.set_planner_id("RRTConnectkConfigDefault")

    group.set_planning_time(0.05)
    
    #int:np.ndarray with transformation matrices
    transformation_matrices = []
    scores = []
    grasp_filters = {}
    filtered_grasps = []

    initial_state = group.get_current_state()

    #If its more than 100 grasps, only take 100, sample randomly
    if len(grasps) > 100:
        rospy.set_param("n_grasps_reduced_to_100", True)
        grasps = np.random.choice(grasps, 100, replace=False)

    
    # Set the logging level to ERROR to suppress warnings

    move_it_filter_grasps = []

    #Loop and display progress bar
    for i,grasp in enumerate(tqdm.tqdm(grasps)):
        # filtered_grasps.append(grasp)
        group.clear_pose_targets()

        group.set_start_state(initial_state)

        success, msg = reach_pose(group, grasp)

        if success:
            filtered_grasps.append(grasp)
            move_it_filter_grasps.append(grasp)

        else:
            grasp.score = -1.0
            move_it_filter_grasps.append(grasp)
        if msg in grasp_filters.keys():
            grasp_filters[msg] += 1
        else:
            #Add new key to dictionary
            grasp_filters[msg] = 1

 


    rospy.set_param("n_grasps_feasible", len(filtered_grasps))
    #Reset the logging level

    chosen_grasps = choice_of_grasps(filtered_grasps)

    rospy.set_param("n_grasps_chosen", len(chosen_grasps))
    
    #New Pc_with_grasps with only the chosen grasps
    new_pc_with_grasps = PointCloudWithGrasps()
    new_pc_with_grasps.point_cloud = pc_msg
    new_pc_with_grasps.grasps = chosen_grasps
    new_pc_with_grasps.header = pc_msg.header

    rospy.loginfo(f"NUMBER OF CHOSEN GRAPHS: {len(chosen_grasps)}")

    # #Delete the octomap
    clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
    clear_octomap()


    pub.publish(new_pc_with_grasps)

    new_pc_with_grasps_moveit = PointCloudWithGrasps()
    new_pc_with_grasps_moveit.point_cloud = pc_msg
    new_pc_with_grasps_moveit.grasps = move_it_filter_grasps
    new_pc_with_grasps_moveit.header = pc_msg.header
    
    pub_moveit_filter.publish(new_pc_with_grasps_moveit)
    
    now = time.time() 
    rospy.set_param('t_filtering_completed', now)


    


def main():
    rospy.init_node('grasp_simulator', anonymous=True)
    rospy.sleep(1)

    pc_publisher = rospy.Publisher('/point_cloud_with_grasps/final', PointCloudWithGrasps, queue_size=30)
    publisher_moveit_filter = rospy.Publisher('/point_cloud_with_grasps/moveit_filter', PointCloudWithGrasps, queue_size=30)
    rospy.Subscriber('/point_cloud_with_grasps/all', PointCloudWithGrasps, callback, callback_args=(pc_publisher, publisher_moveit_filter))

   
    rospy.spin()

if __name__ == '__main__':
    main()