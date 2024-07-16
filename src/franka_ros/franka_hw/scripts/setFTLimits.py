#!/usr/bin/env python3
from typing import List
import rospy
import actionlib
from franka_msgs.srv import SetForceTorqueCollisionBehavior, SetForceTorqueCollisionBehaviorRequest, SetForceTorqueCollisionBehaviorResponse

rospy.init_node("set_ee_frame")

client = rospy.ServiceProxy("/franka_control/set_force_torque_collision_behavior", SetForceTorqueCollisionBehavior)
rospy.loginfo(f"Waiting for service server...")
client.wait_for_service()
rospy.loginfo(f"... service server found")

goal = SetForceTorqueCollisionBehaviorRequest()
goal.lower_force_thresholds_nominal = [-100.0, -100.0, -100.0, -30.0, -30.0, -30.0]
goal.upper_force_thresholds_nominal = [100.0, 100.0, 100.0, 30.0, 30.0, 30.0]
goal.lower_torque_thresholds_nominal = [-70.0, -70.0, -50.0, -50.0, -50.0, -30.0, -30.0]
goal.upper_torque_thresholds_nominal = [70.0, 70.0, 50.0, 50.0, 50.0, 30.0, 30.0]

response : SetForceTorqueCollisionBehaviorResponse =  client.call(goal)
print(f"result:\n{response}")