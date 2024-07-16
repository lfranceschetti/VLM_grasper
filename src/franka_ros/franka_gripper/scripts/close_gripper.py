#!/usr/bin/env python3
from typing import List
import rospy
import actionlib
from geometry_msgs.msg import Point, TransformStamped, Quaternion

from manipulation_msgs.msg import ReachPoseAction, ReachPoseGoal
from franka_gripper.msg import GraspAction, GraspGoal

rospy.init_node("grasp")

gripper_client = actionlib.SimpleActionClient('/franka_gripper/grasp',  GraspAction)
rospy.loginfo(f"Waiting for gripper action server...")
gripper_client.wait_for_server()
rospy.loginfo(f"... gripper action server found")

gripper_goal = GraspGoal()
gripper_goal.speed = 0.01
gripper_goal.width = 0.00
gripper_goal.epsilon.inner = 0.08
gripper_goal.epsilon.outer = 0.08
gripper_goal.force = 50.0
gripper_client.send_goal(gripper_goal)
gripper_client.wait_for_result()
print(f"result:\n{gripper_client.get_result()}")