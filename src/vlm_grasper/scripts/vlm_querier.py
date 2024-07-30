#!/home/lucfra/miniconda3/envs/ros_env/bin/python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import requests
import json
import cv2
import base64
import os
from openai import OpenAI

# Initialize the CvBridge class
bridge = CvBridge()

# Global variables to store the latest image and task description
latest_image = None
latest_task_description = None

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def generate_prompt(task_description, num_choices=4):

    color_choices = ["Green", "Blue", "Red", "Black", "Orange"]
    
    prompt = f"""
    You are a robot Arm. I give you an object that you need to manipulate based on the following task description:

    {task_description}

    Your goal is to find the best grasp position to achieve the given task description. You can assume all the grasps lead to a stable grasp based on the geometric conditions of the object. There are different grasp positions drawn in the image. You have {num_choices} choices to choose from. 
    """

    for i, color in enumerate(color_choices):
        prompt += f"{i+1}. {color}\n"

    prompt += f"""

    Which grasp is the best to achieve the goal?
    """

    return prompt





# Function to query the OpenAI GPT-4 Vision API
def query_openai(image, task_description):
    # Convert the image to a suitable format (e.g., base64 encoding)
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('openaikey')}"
    }

    prompt = generate_prompt(task_description)

    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_base64",
                        "image_base64": image_base64,
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        rospy.logerr(f"Error querying OpenAI API: {response.status_code}, {response.text}")
        return "Error querying OpenAI API"

# Function to trigger OpenAI query if both image and task description are available
def trigger_openai_query():
    if latest_image is not None and latest_task_description is not None:
        best_grasp_position = query_openai(latest_image, latest_task_description)
        rospy.loginfo(f"Best grasp position: {best_grasp_position}")

# Callback function for the image topic
def image_callback(msg):
    global latest_image
    try:
        latest_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        rospy.loginfo("Image received")
        trigger_openai_query()
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge error: {e}")

# Callback function for the task description topic
def task_description_callback(msg):
    global latest_task_description
    latest_task_description = msg.data
    rospy.loginfo("Task description received")
    trigger_openai_query()

def main():
    rospy.init_node('grasp_position_node')

    # Subscribers
    rospy.Subscriber('/camera/color/image_with_grasps', Image, image_callback)
    rospy.Subscriber('/task_instruction', String, task_description_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
