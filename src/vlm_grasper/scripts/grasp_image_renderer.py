#!/home/lucfra/miniconda3/envs/ros_env/bin/python
import cv2
import rospy
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from vlm_grasper.msg import PointCloudWithGrasps
import message_filters
from geometry_msgs.msg import TransformStamped
from vlm_grasper.utils.render_helpers import backproject_point_to_depth_image, create_transformation_matrix, get_grasp_pointclouds, get_grasp_pointclouds
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import open3d as o3d

bridge = CvBridge()

output_dir = os.path.join(os.getcwd(), "saved_images")
os.makedirs(output_dir, exist_ok=True)

latest_camera_info = None

def camera_info_callback(data):
    global latest_camera_info
    latest_camera_info = data


def denormalize_point_cloud(pc):
    points = np.asarray(pc.points)
    
    # Retrieve the stored parameters
    centroid = np.array(rospy.get_param("pc_centroid"))
    factor = rospy.get_param("normalization_factor", 0.15)
    max_dist = rospy.get_param("max_dist")  # Assuming max_dist was also stored
    
    # Reverse scaling by normalization factor
    points /= factor
    
    # Reverse scaling to unit sphere
    points *= max_dist
    
    # Reverse translation to origin
    points += centroid
    
    pc.points = o3d.utility.Vector3dVector(points)
    return pc



def grasp_callback(pc_with_grasps, color_image_msg, depth_image_msg, image_pub):

    rospy.loginfo("Grasps and images received")
    rospy.loginfo("Rendering Image...")

    try:
        depth_image = bridge.imgmsg_to_cv2(depth_image_msg, "16UC1")
        color_image = bridge.imgmsg_to_cv2(color_image_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge error: {e}")
        return
    
    try:
        K_d = np.array(latest_camera_info.K).reshape(3, 3)
        rospy.loginfo(f"Got camera info: {K_d}")
    except AttributeError:
        rospy.logwarn("No camera info received yet")
        return

    #Get point cloud as open3d point cloud
    pc_msg = pc_with_grasps.point_cloud
    gen = pc2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z"))
    points = np.array(list(gen))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)



    #Save image before adding grasps
    try:
        output_path_before = os.path.join(output_dir, "color_image.png")
        cv2.imwrite(output_path_before, color_image)
        rospy.loginfo(f"Saved color image to {output_path_before}")
    except Exception as e:
        rospy.logerr(f"Error saving image: {e}")
        return

    #int:np.ndarray with transformation matrices
    transformation_matrices = []
    scores = []
    for grasp in pc_with_grasps.grasps:
        position = grasp.pose.position
        orientation = grasp.pose.orientation
        # If orientation is quaternion, we assume it's not used and instead use approach and normal
        approach = grasp.approach
        normal = grasp.normal

        score = grasp.score
        scores.append(score)

        matrix = create_transformation_matrix(position, approach, normal)
        transformation_matrices.append(matrix)


    scores = np.array(scores)
    transformation_matrices = np.array(transformation_matrices)
    scores = {0: scores}
    transformation_matrices = {0: transformation_matrices}

    grasp_pointclouds = get_grasp_pointclouds(transformation_matrices, scores)

    rospy.loginfo(f"NUMBER OF GRASP POINTCLOUDS: {len(grasp_pointclouds)}")
    o3d.visualization.draw_geometries([pc] + grasp_pointclouds)



    for grasp_pointcloud in grasp_pointclouds:
        grasp_pointcloud = denormalize_point_cloud(grasp_pointcloud)
        object_pc = denormalize_point_cloud(pc)

    o3d.visualization.draw_geometries([object_pc, grasp_pointcloud])

    #Visualize the point cloud and the grasps


    color_image_with_grasps = color_image.copy()

    changed_pixels = 0

    for grasp_pointcloud in grasp_pointclouds:
        points = np.asarray(grasp_pointcloud.points)
        for point in points:
            
            #Backproject the point to the depth image
            pixel, depth = backproject_point_to_depth_image(point, depth_image, K_d)

            if pixel is not None and depth is not None:
                #If the point is closer to the camera than the current depth value, update the depth value
                if depth < depth_image[pixel[1], pixel[0]]:
                    depth_image[pixel[1], pixel[0]] = depth
                    #Change the color of the pixel in the color image to the color of the point
                    changed_pixels += 1
                    color_image_with_grasps[pixel[1], pixel[0]] = point[:3] * 255

    rospy.loginfo(f"Changed {changed_pixels} pixels in the color image")

    try:
        output_path_after = os.path.join(output_dir, "color_image_grasps.png")
        cv2.imwrite(output_path_after, color_image_with_grasps)
        rospy.loginfo(f"Saved color image with grasps to {output_path_after}")
    except Exception as e:
        rospy.logerr(f"Error saving image: {e}")
        return
    
    try:
        # Publish the processed image with grasps
        image_with_grasps_msg = bridge.cv2_to_imgmsg(color_image_with_grasps, "bgr8")
        image_with_grasps_msg.header = color_image_msg.header
        image_pub.publish(image_with_grasps_msg)
        rospy.loginfo("Published color image with grasps")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge error: {e}")
        return



def main():
    rospy.init_node('grasp_image_renderer')
    
    # Wait for a message to ensure there's data to visualize
    pc_with_grasps = message_filters.Subscriber('/point_cloud_with_grasps', PointCloudWithGrasps)
    color_image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_image_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    camera_info_sub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, camera_info_callback)
    image_pub = rospy.Publisher('/camera/color/image_with_grasps', Image, queue_size=10)



    ts = message_filters.ApproximateTimeSynchronizer([pc_with_grasps, color_image_sub, depth_image_sub], 10, 0.1)
    
    ts.registerCallback(grasp_callback, image_pub)
    
    rospy.spin()


if __name__ == '__main__':
    main()
