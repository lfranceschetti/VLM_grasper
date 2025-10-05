#!/usr/bin/env python

import rospy
import rosbag
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
import roslib
import os

def find_closest_messages(bag, base_msg_time, topics):
    closest_messages = {}
    for topic in topics:
        min_diff = float('inf')
        closest_msg = None
        for _, msg, t in bag.read_messages(topics=[topic]):
            time_diff = abs((t - base_msg_time).to_sec())
            if time_diff < min_diff:
                min_diff = time_diff
                closest_msg = msg
        closest_messages[topic] = closest_msg
    return closest_messages

def load_bag_file(scene):
    package_path = roslib.packages.get_pkg_dir('vlm_grasper')
    bag_path = package_path + "/experiments/"+ scene + "/" + scene + '.bag'

    rospy.loginfo(f"Opening bag file: {bag_path}")
    
    # Open the bag file
    try:
        bag = rosbag.Bag(bag_path, 'r')
    except Exception as e:
        rospy.logfatal(f"Failed to open bag file: {bag_path}. Error: {e}")
        return None
    
    return bag

def get_initial_messages(bag, base_topic, topics_to_sync):
    # Retrieve the first PointCloud2 message from the base topic
    base_msg = None
    base_msg_time = None
    for topic, msg, t in bag.read_messages(topics=[base_topic]):
        base_msg = msg
        base_msg_time = t
        break


    if not base_msg:
        rospy.logwarn("No messages found on the base topic. Retrying in 2 seconds...")
        rospy.sleep(2)
        return get_initial_messages(bag, base_topic, topics_to_sync)

    # Find the closest messages in time for the other topics
    closest_messages = find_closest_messages(bag, base_msg_time, topics_to_sync)
    
    return base_msg, closest_messages

def main():
    rospy.init_node('bag_publisher', anonymous=True)

    current_scene = rospy.get_param('scene', '')
    if current_scene == '':
        rospy.logfatal("Scene parameter not set. Please set the 'scene' parameter.")
        return

    bag = load_bag_file(current_scene)
    if not bag:
        return

    base_topic = '/camera/depth/color/points'
    topics_to_sync = [
        '/camera/color/camera_info',
        '/camera/aligned_depth_to_color/camera_info',
        '/camera/color/image_raw',
        '/camera/aligned_depth_to_color/image_raw'
    ]

    base_msg, closest_messages = get_initial_messages(bag, base_topic, topics_to_sync)
    if not base_msg or not closest_messages:
        bag.close()
        return

    # Publishers for each topic
    point_cloud_publisher = rospy.Publisher(base_topic, PointCloud2, queue_size=2, latch=True)
    camera_info_color_publisher = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=2, latch=True)
    camera_info_depth_publisher = rospy.Publisher('/camera/aligned_depth_to_color/camera_info', CameraInfo, queue_size=2, latch=True)
    image_color_publisher = rospy.Publisher('/camera/color/image_raw', Image, queue_size=2, latch=True)
    image_depth_aligned_publisher = rospy.Publisher('/camera/aligned_depth_to_color/image_raw', Image, queue_size=2, latch=True)

    rospy.sleep(1.0)  # Give some time for other nodes to connect

    rate = rospy.Rate(0.06)  # Define the rate at which to publish messages

    while not rospy.is_shutdown():
        # Monitor the scene parameter for changes
        new_scene = rospy.get_param('scene', '')
        if new_scene != current_scene:
            rospy.loginfo(f"Scene parameter changed from {current_scene} to {new_scene}. Loading new scene.")
            bag.close()

            current_scene = new_scene
            bag = load_bag_file(current_scene)
            if not bag:
                break

            base_msg, closest_messages = get_initial_messages(bag, base_topic, topics_to_sync)
            if not base_msg or not closest_messages:
                bag.close()
                break

        now = rospy.Time.now()
        base_msg.header.stamp = now
        closest_messages['/camera/color/camera_info'].header.stamp = now
        closest_messages['/camera/aligned_depth_to_color/camera_info'].header.stamp = now
        closest_messages['/camera/color/image_raw'].header.stamp = now
        closest_messages['/camera/aligned_depth_to_color/image_raw'].header.stamp = now

        # Publish the messages
        point_cloud_publisher.publish(base_msg)
        camera_info_color_publisher.publish(closest_messages['/camera/color/camera_info'])
        camera_info_depth_publisher.publish(closest_messages['/camera/aligned_depth_to_color/camera_info'])
        image_color_publisher.publish(closest_messages['/camera/color/image_raw'])
        image_depth_aligned_publisher.publish(closest_messages['/camera/aligned_depth_to_color/image_raw'])

        rate.sleep()

    if bag:
        bag.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
