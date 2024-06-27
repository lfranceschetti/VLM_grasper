#!/home/lucfra/miniconda3/envs/ros_env/bin/python

import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os

# Initialize CvBridge
bridge = CvBridge()

# Path to your bag file
bag_file = 'plant4.bag'

image_sources = ['/camera/color/image_raw']

for image_source in image_sources:
    # Output directory for images
    output_dir = bag_file.split('.')[0] + "/" + "images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the bag file
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[image_source]):
            try:
                print(f'Processing message from topic: {topic}, timestamp: {t.to_nsec()}')

                # Check the encoding of the incoming image
                if msg.encoding == "16UC1":
                    # Handle depth image
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                    print("Hi")
                    # Normalize the depth image to 8-bit for visualization
                    cv_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
                    cv_image_8u = cv_image_normalized.astype('uint8')
                    # Optionally convert to color for visualization
                    cv_image_color = cv2.applyColorMap(cv_image_8u, cv2.COLORMAP_JET)
                    # Save the image to a file
                    image_file = os.path.join(output_dir, 'depth_image_{}.png'.format(t.to_nsec()))
                    cv2.imwrite(image_file, cv_image_color)
                else:
                    # Handle other image types
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    # Save the image to a file
                    image_file = os.path.join(output_dir, 'image_{}.png'.format(t.to_nsec()))
                    cv2.imwrite(image_file, cv_image)

                print(f'Saved image: {image_file}')

            except CvBridgeError as e:
                print(f'CvBridge Error: {e}')
            except Exception as e:
                print(f'An error occurred: {e}')

print('Finished processing all messages.')
