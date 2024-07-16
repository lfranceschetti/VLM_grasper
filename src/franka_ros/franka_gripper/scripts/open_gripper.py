#!/usr/bin/env python3
import rospy
import actionlib
import argparse
from geometry_msgs.msg import Point, TransformStamped, Quaternion

from manipulation_msgs.msg import ReachPoseAction, ReachPoseGoal
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspGoal

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--width", type=float, default=2, help="opening width in cm")
args = parser.parse_args()


rospy.init_node("open")

gripper_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
rospy.loginfo(f"Waiting for gripper action server...")
gripper_client.wait_for_server()
rospy.loginfo(f"... gripper action server found")

gripper_goal = MoveGoal()
gripper_goal.speed = 0.1
gripper_goal.width = args.width / 100
gripper_client.send_goal(gripper_goal)
gripper_client.wait_for_result()
print(f"result:\n{gripper_client.get_result()}")
