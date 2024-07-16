import rosbag

# Path to your bag file
bag_file = 'plant4.bag'
camera_info_topic = {"Color": '/camera/color/camera_info', "Depth": '/camera/depth/camera_info'}


for name, cam_topic in camera_info_topic.items():
    # Open the bag file and read the camera info
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[cam_topic]):
            print(f'Camera info message for {name} camera:')
            print(msg)
            # Extract the intrinsic parameters
            fx = msg.K[0]  # Focal length in x
            fy = msg.K[4]  # Focal length in y
            cx = msg.K[2]  # Optical center in x
            cy = msg.K[5]  # Optical center in y
            break  # Assuming all camera info messages are the same, we can break after the first one


with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/tf_static']):
        print(msg)