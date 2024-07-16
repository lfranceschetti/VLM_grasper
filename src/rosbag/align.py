import pyrealsense2 as rs
import numpy as np
import cv2

# Path to the .bag file
bag_file_path = "plant4.bag"

# Configure depth and color streams
config = rs.config()
config.enable_device_from_file(bag_file_path)
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming from file
pipeline = rs.pipeline()
profile = pipeline.start(config)

# Align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale is: {depth_scale}")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        # Get intrinsics of the depth frame
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Display images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)

        # Break loop on 'ESC' key press
        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()
