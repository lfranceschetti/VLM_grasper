#!/usr/bin/env python
import cv2
import rospy
import os
import numpy as np
import open3d as o3d
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from vlm_grasper.msg import PointCloudWithGrasps
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import time
import roslib
from vlm_grasper.utils.render_helpers import backproject_point_to_depth_image, create_transformation_matrix, get_grasp_pointclouds, get_grasp_color

class GraspImageRenderer:
    def __init__(self):
        rospy.init_node('grasp_image_renderer')

        self.bridge = CvBridge()
        self.latest_camera_info = None

        # Initialize storage for incoming messages
        self.pc_w_grasps_final = None
        self.grasps_filtered_score = None
        self.grasps_filtered_moveit = None
        self.color_image_msg = None
        self.depth_image_msg = None
        self.original_pc = None

        self.last_publish_time = rospy.Time.now()


        # Output directory for saving images
        self.output_dir = os.path.join(os.getcwd(), "saved_images")
        os.makedirs(self.output_dir, exist_ok=True)

        # ROS Subscribers
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/point_cloud_with_grasps/final', PointCloudWithGrasps, self.pc_w_grasps_final_callback)
        rospy.Subscriber('/point_cloud_with_grasps/score_filter', PointCloudWithGrasps, self.grasps_filtered_score_callback)
        rospy.Subscriber('/point_cloud_with_grasps/moveit_filter', PointCloudWithGrasps, self.grasps_filtered_moveit_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_image_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_image_callback)
        rospy.Subscriber('/point_cloud/objects', PointCloud2, self.original_pc_callback)

        # ROS Publisher
        self.image_pub = rospy.Publisher('/rendered_image/image_with_grasps', Image, queue_size=10)

    def camera_info_callback(self, data):
        self.latest_camera_info = data

    def pc_w_grasps_final_callback(self, msg):
        self.pc_w_grasps_final = msg
        self.try_render_final_image()

    def grasps_filtered_score_callback(self, msg):
        self.grasps_filtered_score = msg
        self.try_render_score_filter_image()

    def grasps_filtered_moveit_callback(self, msg):
        self.grasps_filtered_moveit = msg
        self.try_render_moveit_filter_image()

    def color_image_callback(self, msg):
        self.color_image_msg = msg
        self.try_render_final_image()
        self.try_render_score_filter_image()
        self.try_render_moveit_filter_image()

    def depth_image_callback(self, msg):
        self.depth_image_msg = msg
        self.try_render_final_image()
        self.try_render_score_filter_image()
        self.try_render_moveit_filter_image()

    def original_pc_callback(self, msg):
        self.original_pc = msg
        self.try_render_final_image()
        self.try_render_score_filter_image()
        self.try_render_moveit_filter_image()

    def get_pc_from_msg(self, msg):
        gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        points = np.array(list(gen))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        return pc

    def render_image_with_grasps(self, pc_with_grasps, color_image_msg, depth_image_msg, original_pc, render_original_image_path=None, method=None):

        if rospy.get_param("grasp_image_renderer_finished", False):
            return None
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "16UC1")
            color_image = self.bridge.imgmsg_to_cv2(color_image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")
            return None

        try:
            K_d = np.array(self.latest_camera_info.K).reshape(3, 3)
        except AttributeError:
            rospy.logwarn("No camera info received yet")
            return None

        # Save original image if path is provided
        if render_original_image_path:
            try:
                cv2.imwrite(render_original_image_path, color_image)
            except Exception as e:
                rospy.logerr(f"Error saving image: {e}")
                return None

        transformation_matrices = []
        scores = []

        rospy.set_param("n_final_grasps", len(pc_with_grasps.grasps))

        for grasp in pc_with_grasps.grasps:
            matrix = create_transformation_matrix(grasp.pose.position, grasp.approach, grasp.normal)
            transformation_matrices.append(matrix)
            scores.append(grasp.score)

        scores = np.array(scores)
        grasp_pointclouds = get_grasp_pointclouds(np.array(transformation_matrices), scores)
        original_pc = self.get_pc_from_msg(original_pc)

        color_image_with_grasps = color_image.copy()

        # Create mask of object
        object_mask = np.zeros_like(depth_image)
        for object_point in original_pc.points:
            pixel, depth = backproject_point_to_depth_image(object_point, depth_image, K_d)
            if pixel is not None and depth is not None:
                depth_image[pixel[1], pixel[0]] = depth
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if abs(i + j) <= 2:
                            object_mask[pixel[1] + i, pixel[0] + j] = 1

        filter_colors = -1.0 in scores
        changed_pixels = 0

        for i, grasp_pointcloud in enumerate(grasp_pointclouds):
            points = np.asarray(grasp_pointcloud.points)
            color = np.array(get_grasp_color(i)) * 255

            if filter_colors and scores[i] != -1.0:
                if method == "all_grey":
                    color = [0, 0, 0]
                elif method == "red_green":
                    color = [0, 255, 0]
                elif method == "green":
                    color = [0, 255, 0]
            if filter_colors and scores[i] == -1.0:
                if method == "all_grey":
                    color = [0, 0, 0]
                elif method == "red_green":
                    color = [0, 0, 255]
                elif method == "green":
                    continue

            for point in points:
                pixel, depth = backproject_point_to_depth_image(point, depth_image, K_d)
                if pixel is not None and depth is not None:
                    if depth < depth_image[pixel[1], pixel[0]] or object_mask[pixel[1], pixel[0]] == 0:
                        for i in range(-1, 2):
                            for j in range(-1, 1):
                                if abs(i) + abs(j) <= 1:
                                    depth_image[pixel[1] + i, pixel[0]+j] = depth
                                    changed_pixels += 1
                                    color_image_with_grasps[pixel[1]+i, pixel[0]+j] = color

        return color_image_with_grasps

    def try_render_final_image(self):
        if self.pc_w_grasps_final and self.color_image_msg and self.depth_image_msg and self.original_pc:
            results_path = os.path.join(roslib.packages.get_pkg_dir('vlm_grasper'), "experiments", rospy.get_param('scene', 'unknown'), "results")
            image_with_grasps = self.render_image_with_grasps(self.pc_w_grasps_final, self.color_image_msg, self.depth_image_msg, self.original_pc, render_original_image_path=os.path.join(results_path, "color_image.png"))

            image_name = f"color_image_grasps_{rospy.get_param('experiment_number', 0)}.png"
            # image_name = "color_image_grasps.png"
            if image_with_grasps is not None:
                self.save_and_publish_image(image_with_grasps, os.path.join(results_path, image_name), "Published color image with grasps")

    def try_render_score_filter_image(self):
        if self.grasps_filtered_score and self.color_image_msg and self.depth_image_msg and self.original_pc:
            results_path = os.path.join(roslib.packages.get_pkg_dir('vlm_grasper'), "experiments", rospy.get_param('scene', 'unknown'), "results")
            if not os.path.exists(results_path):
                os.makedirs(results_path)
        
            score_filter_image2 = self.render_image_with_grasps(self.grasps_filtered_score, self.color_image_msg, self.depth_image_msg, self.original_pc, method="red_green")
            if score_filter_image2 is not None:
                self.save_image(score_filter_image2, os.path.join(results_path, "color_image_grasps_score_filter.png"))

    def try_render_moveit_filter_image(self):
        if self.grasps_filtered_moveit and self.color_image_msg and self.depth_image_msg and self.original_pc:
            results_path = os.path.join(roslib.packages.get_pkg_dir('vlm_grasper'), "experiments", rospy.get_param('scene', 'unknown'), "results")
            moveit_filter_image = self.render_image_with_grasps(self.grasps_filtered_moveit, self.color_image_msg, self.depth_image_msg, self.original_pc, method="red_green")
          
            if moveit_filter_image is not None:
                self.save_image(moveit_filter_image, os.path.join(results_path, "color_image_grasps_moveit_filter.png"))

    def save_and_publish_image(self, image, path, log_message):
        current_time = rospy.Time.now()
        time_since_last_publish = (current_time - self.last_publish_time).to_sec()


        if time_since_last_publish < 1.0:
            rospy.loginfo("Skipping publish to maintain 1-second interval")
            return
        
        try:
            cv2.imwrite(path, image)
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            image_msg.header = self.color_image_msg.header
            self.image_pub.publish(image_msg)
            rospy.loginfo(log_message)
            now = rospy.Time.now()
            self.last_publish_time = now  # Update last publish time

            time_now = time.time() 

            rospy.set_param('t_rendering_completed', time_now)

            self.reset_renderer()
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error saving or publishing image: {e}")

    def save_image(self, image, path):
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            rospy.logerr(f"Error saving image: {e}")

    def reset_renderer(self):
        self.pc_w_grasps_final = None
        self.grasps_filtered_score = None
        self.grasps_filtered_moveit = None
        self.color_image_msg = None
        self.depth_image_msg = None
        self.original_pc = None

if __name__ == '__main__':
    GraspImageRenderer()
    rospy.spin()
