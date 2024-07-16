#!/usr/bin/env python3
from typing import List
import rospy
import actionlib
from franka_msgs.srv import SetLoad, SetLoadRequest, SetLoadResponse

rospy.init_node("set_load")

client = rospy.ServiceProxy("/franka_control/set_load", SetLoad)
rospy.loginfo(f"Waiting for service server...")
client.wait_for_service()
rospy.loginfo(f"... service server found")

goal = SetLoadRequest()
goal.mass = 0.25 # mass of disk object
goal.F_x_center_load = [0.0, 0.0, 0.0]
goal.load_inertia = [ 0.00001308, 0, 0, 0, 0.00001875, 0, 0, 0, 0.00000783 ]
response : SetLoadResponse =  client.call(goal)
print(f"result:\n{response}")