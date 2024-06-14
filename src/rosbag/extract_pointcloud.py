#!/usr/bin/env python

import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import struct
from sensor_msgs.msg import PointCloud2

def convert_rgb_float_to_tuple(rgb_float):
    rgb = struct.unpack('I', struct.pack('f', rgb_float))[0]
    r = (rgb >> 16) & 0x0000ff
    g = (rgb >> 8) & 0x0000ff
    b = (rgb) & 0x0000ff
    return (r, g, b)

# Path to your bag file
bag_file = 'cam_capture_metal_cup.bag'

# Open the bag file
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/camera/depth/color/points']):
        # Convert ROS PointCloud2 message to a list of points
        points_list = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb = point
            r, g, b = convert_rgb_float_to_tuple(rgb)
            points_list.append([x, y, z, r/255.0, g/255.0, b/255.0])
        
        # Create an Open3D point cloud
        points_array = np.array(points_list)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_array[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points_array[:, 3:])
        
        # Visualize the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd)
        
        # Capture and save the image
        vis.poll_events()
        vis.update_renderer()

        vis.run()
        # vis.capture_screen_image(f"extracted_pcs/{bag_file.split('.')[0]}/point_cloud_{t.to_nsec()}.png")
        # Close the visualizer
        # vis.destroy_window()

print("PointCloud visualization complete and images saved.")
