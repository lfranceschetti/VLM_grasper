#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import yaml
from vlm_grasper.msg import TaskArray  # Import the custom message
import os
import roslib


def load_tasks_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['tasks']

def main():
    rospy.init_node('task_publisher_node')

    task_publisher = rospy.Publisher('/task_instruction', TaskArray, queue_size=10)
    
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        
        tasks = rospy.get_param('tasks', [])
        task_msg = TaskArray()
        if len(tasks) > 0:
            task_msg.tasks = tasks  # Array of tasks loaded from YAML
            task_publisher.publish(task_msg)
        rate.sleep()

if __name__ == '__main__':
    main()