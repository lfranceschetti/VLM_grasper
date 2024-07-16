#!/home/lucfra/miniconda3/envs/ros_env/bin/python

import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import numpy as np

# Initialize CvBridge
bridge = CvBridge()

# Path to your bag file
bag_file = 'pingu2.bag'

image_sources = ['/camera/aligned_depth_to_color/image_raw', '/camera/color/image_raw']
output_dir = bag_file.split('.')[0] + "/" + "depth_images/npy"


def save_depth_image(msg, t, method="image"):
    if method=="image":
         # Handle depth image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        print("min: ", np.min(cv_image))
        print("max: ", np.max(cv_image))

        # Normalize the depth image to 8-bit for visualization
        cv_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
        cv_image_8u = cv_image_normalized.astype('uint8')
        #Make the difference between the min and max values of the depth image
        

        # Optionally convert to color for visualization
        cv_image_color = cv2.applyColorMap(cv_image_8u, cv2.COLORMAP_JET)
        # Save the image to a file
        image_file = os.path.join(output_dir, 'depth_image_{}.png'.format(t.to_nsec()))
        cv2.imwrite(image_file, cv_image_color)

    elif method=="npy":
        # Handle depth image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # Save the image to a file
        image_file = os.path.join(output_dir, 'depth_image_{}.npy'.format(t.to_nsec()))
        np.save(image_file, cv_image)
        


for image_source in image_sources:
    # Output directory for images

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the bag file
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[image_source]):
            try:
                print(f'Processing message from topic: {topic}, timestamp: {t.to_nsec()}')

                # Check the encoding of the incoming image
                if msg.encoding == '16UC1':
                    save_depth_image(msg,t, method="npy")
                    save_depth_image(msg,t, method="image")
                
                else:
                    # Handle other image types
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                    # Save the image to a file
                    image_file = os.path.join(output_dir, 'image_{}.png'.format(t.to_nsec()))
                    cv2.imwrite(image_file, cv_image)


            except CvBridgeError as e:
                print(f'CvBridge Error: {e}')
            except Exception as e:
                print(f'An error occurred: {e}')

print('Finished processing all messages.')
