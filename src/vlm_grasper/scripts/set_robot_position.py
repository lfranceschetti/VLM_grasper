#!/home/lucfra/miniconda3/envs/ros_env/bin/python
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
import sys

class GraspPipeline:
    def __init__(self):
        rospy.init_node('grasp_pipeline', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize MoveIt Commander
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_planning_time(10)

        # Set up a short delay to ensure MoveGroup is initialized
        rospy.sleep(5)
        self.move_to_initial_position()


        # Subscribe to the grasp pose topic
        rospy.Subscriber('/grasp_pose', Pose, self.grasp_callback)

    def move_to_initial_position(self):
        # Define the initial pose

        initial_joint_values = [0.000, -1.220, 0.0, -1.220, 0.0, 0.954, 0.787]

        self.group.set_joint_value_target(initial_joint_values)

        # Plan and execute the movement
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        if plan:
            rospy.loginfo("Moved to initial position successfully.")
        else:
            rospy.logwarn("Failed to move to initial position.")

    def grasp_callback(self, pose):
        # Set the target pose for the end-effector
        self.group.set_pose_target(pose)

        # Plan and execute the grasp
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        if plan:
            rospy.loginfo("Grasp executed successfully.")
        else:
            rospy.logwarn("Failed to execute grasp.")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    grasp_pipeline = GraspPipeline()
    grasp_pipeline.run()
