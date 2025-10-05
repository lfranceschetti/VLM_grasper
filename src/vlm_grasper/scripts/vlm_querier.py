#!/usr/bin/env python

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
from vlm_grasper.msg import TaskArray  # Import the custom message
import roslib
import time

# Initialize the CvBridge class
bridge = CvBridge()

# Global variables to store the latest image and task instructions
latest_image = None
latest_task_descriptions = None

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def save_results(results, overwrite=True):
    results_path = roslib.packages.get_pkg_dir('vlm_grasper') + "/experiments/" + rospy.get_param('scene', 'unknown') + "/results"
    results_file = os.path.join(results_path, "results.json")

    if not overwrite and os.path.exists(results_file):
        with open(results_file, 'r') as file:
            existing_results = json.load(file)
    else:
        existing_results = []

    # Append new results to existing results
    existing_results.extend(results)

    with open(results_file, 'w') as file:
        file.write(json.dumps(existing_results, indent=4))

    global latest_image
    latest_image = None

    global latest_task_descriptions
    latest_task_descriptions = None

    now = time.time() 
    rospy.set_param("t_end", now)

    rospy.set_param("finished", False)
    rospy.set_param("finished", True)







def generate_prompt(task_description, num_choices=4):

    color_choices = ["Green", "Red", "Blue", "Black", "White"]
    
    # Ensure num_choices does not exceed the number of available colors
    selected_colors = color_choices[:num_choices]
    
    # Create the reasoning dictionary template
    reasoning_dict = "\n".join(
        [f'"{color}": "Provide reasoning for the {color} grasp here."' for color in selected_colors]
    )
    
    prompt = f"""
    **Task Description:**

    You are a robot arm equipped with advanced reasoning capabilities. Your task is to pick up an object based on the following instructions:

    - **Goal:** {task_description}
    - **Context:** You have identified {num_choices} potential grasp positions, each marked with a different color: {", ".join(selected_colors)}. Each grasp is depicted as a "T" shape where the vertical line represents the central approach direction of the gripper. The horizontal line at the top of the "T" represents the base of the gripper. Two additional short parallel lines are placed near the ends of the horizontal line, extending outward parallel to the approach. These lines represent the fingers of the gripper that would close around the object.

    **Step 1: Scene Description**
    Begin by describing the scene in detail. Include the location and orientation of the object(s) relative to your current position. Only focus on what is visible on the table. For this ignore the grasps themselves. For single objects also closely describe the object itself and especially the orientation of the object.

    **Step 2: Evaluate Task Instructions**
    After describing the scene, evaluate which objects in the scene can be used to fulfill the task instructions. Also its very important to describe where would you grasp the object (which part) to fulfill the instruction as well as possible and where this part is w.r.t the object (top/bottom/left/right). If there is no specific part describe the area (top,left,right,front,back of the object). Also include approaches that make sense. So if a single object is oriented in a way that it should be grasped from the top to fulfill the task, describe that. If the object is oriented in a way that it should be grasped from the side, describe that.
    Also make sure the robot doesnt get dirty, damaged or the object is damaged. 

    **Step 3: Reasoning for Each Grasp**
    After thinking about how to achieve the task, evaluate each grasp position. For each color, provide reasoning about its suitability for the task. Consider factors such as orientation, object shape, and alignment with the goal.

    **Step 4: Final Decision**
    After considering all options, select the grasp position that best fits the task. If multiple positions seem equally plausible, choose the one that has a slight advantage in achieving the goal. If none of the positions seem suitable, respond with "None".

    **Output Format:**
    Provide your response in the following structured format:

    {{
        "scene_description": "Provide a detailed description of the scene here."
        ,
        "task_evaluation": "Provide an evaluation of the task instructions here."
        ,
        "reasoning": {{
            {reasoning_dict}
        }}
        ,
        "choice": "The color that corresponds to the best grasp position as String (e.g., \\"Red\\". If no suitable option exists, respond with \\"None\\"."
    }}
    Give it as a string in JSON format and not as a JSON directly.Make sure to add commas between the different parts of objects (especially between the reasoning objects).
    """

    return prompt




# Function to query the OpenAI GPT-4 Vision API
def query_openai(image, task_description):
    # Convert the image to a suitable format (e.g., base64 encoding)
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    number_of_grasps_found = rospy.get_param("n_grasps_chosen", 5)

    prompt = generate_prompt(task_description, num_choices=number_of_grasps_found)

    data = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        }
                    },
                ],
            }
        ],
        "max_tokens": 1500,
    }

    # Perform the POST request to OpenAI API
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()



            return response_data['choices'][0]['message']['content']
        else:


            rospy.logerr(f"Error querying OpenAI API: {response.status_code}, {response.text}")
            return "Error querying OpenAI API"
    except requests.exceptions.RequestException as e:
        rospy.logerr(f"Request to OpenAI API failed: {e}")
        return "Failed to query OpenAI API"

# Function to trigger OpenAI query if both image and task instructions are available
def trigger_openai_query():
    if latest_image is not None and latest_task_descriptions is not None and len(latest_task_descriptions) > 0:
        finished = rospy.get_param("finished", False)

        if finished:
            rospy.loginfo("Already finished processing the task instructions")
            return

        results = []

        for i, task in enumerate(latest_task_descriptions):

            rospy.loginfo(f"Querying OpenAI for task: {task}")

            max_attempts = 5
            attempts = 0
            response_data = None
            while attempts < max_attempts:
                openai_response = query_openai(latest_image, task)

                # Start after the first { and end before the last }
                openai_response = "{" + openai_response[openai_response.find("{") + 1:openai_response.rfind("}")] + "}"
                rospy.loginfo(f"OpenAI response: {openai_response}")
                
                try:
                    response_data = json.loads(openai_response)
                    break  # Exit the loop if parsing is successful
                except json.JSONDecodeError as e:
                    attempts += 1
                    rospy.logerr(f"Error parsing OpenAI response (attempt {attempts}): {e}")

                    if attempts >= max_attempts:
                        rospy.logerr("Max attempts reached. Unable to parse OpenAI response.")
                    else:
                        rospy.loginfo("Retrying...")

            if response_data is None:
                rospy.logerr("Failed to parse OpenAI response. Skipping task.")
                continue

            results.append({
                "task": task,
                "best_grasp_position": response_data['choice'],
                "reasoning": response_data['reasoning'],
                "scene_description": response_data['scene_description'],
                "task_evaluation": response_data['task_evaluation'],
                "experiment_number": rospy.get_param("experiment_number", 0),
            })

            rospy.loginfo(f"Task {i + 1} processed")
            rospy.loginfo(f"Best grasp position: {response_data['choice']}")

        save_results(results, overwrite=False)

# Callback function for the image topic
def image_callback(msg):
    global latest_image
    rospy.set_param("grasp_image_renderer_finished", True)
    try:
        latest_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        rospy.loginfo("Image received")
        trigger_openai_query()
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge error: {e}")

# Callback function for the task instructions topic
def task_description_callback(msg):
    global latest_task_descriptions
    latest_task_descriptions = msg.tasks

def main():
    rospy.init_node('vlm_querier')

    # Subscribers
    rospy.Subscriber('/rendered_image/image_with_grasps', Image, image_callback)
    rospy.Subscriber('/task_instruction', TaskArray, task_description_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
