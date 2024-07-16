#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np
import copy

from interactive_markers.interactive_marker_server import \
    InteractiveMarkerServer, InteractiveMarkerFeedback
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import InteractiveMarker, \
    InteractiveMarkerControl, Marker
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState
from manipulation_msgs.msg import ReachPoseActionGoal

marker_pose = PoseStamped()
initial_pose_found = False
pose_pub = None
goal_pose_pub = None
# [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
position_limits = [[-0.6, 0.6], [-0.6, 0.6], [-0.05, 0.9]]


def publisher_callback():
    marker_pose.header.frame_id = link_name
    marker_pose.header.stamp = rospy.Time(0)
    pose_pub.publish(marker_pose)


def franka_state_callback(msg):
    initial_quaternion = \
        tf.transformations.quaternion_from_matrix(
            np.transpose(np.reshape(msg.O_T_EE,
                                    (4, 4))))
    initial_quaternion = initial_quaternion / \
        np.linalg.norm(initial_quaternion)
    marker_pose.pose.orientation.x = initial_quaternion[0]
    marker_pose.pose.orientation.y = initial_quaternion[1]
    marker_pose.pose.orientation.z = initial_quaternion[2]
    marker_pose.pose.orientation.w = initial_quaternion[3]
    marker_pose.pose.position.x = msg.O_T_EE[12]
    marker_pose.pose.position.y = msg.O_T_EE[13]
    marker_pose.pose.position.z = msg.O_T_EE[14]
    global initial_pose_found
    initial_pose_found = True


def process_feedback(feedback):
    if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        marker_pose.pose.position.x = max([min([feedback.pose.position.x,
                                          position_limits[0][1]]),
                                          position_limits[0][0]])
        marker_pose.pose.position.y = max([min([feedback.pose.position.y,
                                          position_limits[1][1]]),
                                          position_limits[1][0]])
        marker_pose.pose.position.z = max([min([feedback.pose.position.z,
                                          position_limits[2][1]]),
                                          position_limits[2][0]])
        marker_pose.pose.orientation = feedback.pose.orientation
    elif feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
        publisher_callback()

    server.applyChanges()

def menu_feedback(feedback : InteractiveMarkerFeedback):
    if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
        goal = ReachPoseActionGoal()
        marker_pose.header.frame_id = link_name
        marker_pose.header.stamp = rospy.Time.now()
        goal.goal.poseStamped = marker_pose
        goal_pose_pub.publish(goal)

    server.applyChanges()

def makeBox(msg):
    marker = Marker()

    marker.type = Marker.SPHERE
    marker.scale.x = msg.scale * 0.45
    marker.scale.y = msg.scale * 0.45
    marker.scale.z = msg.scale * 0.45
    marker.color.r = 0.0
    marker.color.g = 0.5
    marker.color.b = 0.5
    marker.color.a = 0.5
    return marker


if __name__ == "__main__":
    rospy.init_node("equilibrium_pose_node")
    state_sub = rospy.Subscriber("franka_state_controller/franka_states",
                                 FrankaState, franka_state_callback)
    listener = tf.TransformListener()
    link_name = rospy.get_param("~link_name")

    # Get initial pose for the interactive marker
    while not initial_pose_found and not rospy.is_shutdown():
        rospy.sleep(1)
    state_sub.unregister()

    pose_pub = rospy.Publisher(
        "equilibrium_pose", PoseStamped, queue_size=10)

    goal_pose_pub = rospy.Publisher(
        "/task_space_action_server/task_space/goal", ReachPoseActionGoal, queue_size=1, latch=True)

    server = InteractiveMarkerServer("equilibrium_pose_marker")
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = link_name
    int_marker.scale = 0.3
    int_marker.name = "equilibrium_pose"
    int_marker.description = ("Equilibrium Pose\nBE CAREFUL! "
                              "If you move the \nequilibrium "
                              "pose the robot will follow it\n"
                              "so be aware of potential collisions")
    int_marker.pose = marker_pose.pose
    # run pose publisher
    # rospy.Timer(rospy.Duration(0.005),
    #             lambda msg: publisher_callback(msg, link_name))

    # insert a box
    control = InteractiveMarkerControl()
    control.orientation.w = 0.707
    control.orientation.x = 0.707
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "rotate_x"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    int_marker.controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 0.707
    control.orientation.x = 0.707
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "move_x"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)
    control = InteractiveMarkerControl()
    control.orientation.w = 0.707
    control.orientation.x = 0
    control.orientation.y = 0.707
    control.orientation.z = 0
    control.name = "rotate_y"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    int_marker.controls.append(control)
    control = InteractiveMarkerControl()
    control.orientation.w = 0.707
    control.orientation.x = 0
    control.orientation.y = 0.707
    control.orientation.z = 0
    control.name = "move_y"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)
    control = InteractiveMarkerControl()
    control.orientation.w = 0.707
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 0.707
    control.name = "rotate_z"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    int_marker.controls.append(control)
    control = InteractiveMarkerControl()
    control.orientation.w = 0.707
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 0.707
    control.name = "move_z"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)

    control = InteractiveMarkerControl()
    control.interaction_mode = InteractiveMarkerControl.BUTTON
    control.name = "click"
    control.markers.append(makeBox(int_marker))
    int_marker.controls.append(copy.deepcopy(control))

    server.insert(int_marker, process_feedback)

    # apply right-click menu
    menu_handler = MenuHandler()
    menu_handler.insert("Send to action server", callback=menu_feedback)
    menu_handler.apply(server, "equilibrium_pose")

    server.applyChanges()

    rospy.spin()
