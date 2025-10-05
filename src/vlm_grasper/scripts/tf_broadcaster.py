#!/usr/bin/env python

import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import Vector3Stamped, TransformStamped
import message_filters
from scipy.spatial.transform import Rotation as R


from tf.transformations import quaternion_from_matrix, translation_from_matrix

def calculate_tsc(surface_normal, center_of_pc):
    # Assuming the scene frame has Z pointing outwards as (0, 0, 1)
    z_scene = np.array([0, 0, 1])
    
    # Surface normal should be unit length
    z_camera = np.array([surface_normal.vector.x, surface_normal.vector.y, surface_normal.vector.z])
    z_camera = z_camera / np.linalg.norm(z_camera)

    y_scene = np.array([0, 1, 0])
    y_camera = np.array([-1, 0, 0]) 
    
    
    rotation, _ = R.align_vectors([z_scene, y_scene], [z_camera, y_camera])
    rotation = rotation.as_matrix()
    
    # Translation: center_of_pc should be translated to (0, 0, 0) in scene frame
    translation = np.array([center_of_pc.vector.x, center_of_pc.vector.y, center_of_pc.vector.z])
    
    # Build the 4x4 transformation matrix Tsc
    Tsc = np.eye(4)
    Tsc[0:3, 0:3] = rotation
    Tsc[0:3, 3] = translation

    
    return Tsc

def calculate_twc(Tws, Tsc):
    # Calculate Twc = Tws * Tsc
    return np.dot(Tws, Tsc)


def calculate_callback(surface_normal_msg, center_of_pc_msg):

    # Calculate Tsc
    Tsc = calculate_tsc(surface_normal_msg, center_of_pc_msg)
    
    # Calculate the resulting transform Twc
    Twc = calculate_twc(Tws, Tsc)

    stamp = surface_normal_msg.header.stamp
    
    # Broadcast the Twc transformation
    broadcast_transform(Twc, stamp)

def broadcast_transform(Twc, stamp):
    transform = TransformStamped()

    transform.header.stamp = stamp
    transform.header.frame_id = "world"
    transform.child_frame_id = "camera_color_optical_frame"
    
    # Extract translation
    transform.transform.translation.x = Twc[0, 3]
    transform.transform.translation.y = Twc[1, 3]
    transform.transform.translation.z = Twc[2, 3]

    # Extract rotation and convert to quaternion
    rotation = quaternion_from_matrix(Twc)

    norm = np.linalg.norm(rotation)
    rotation = rotation / norm

    transform.transform.rotation.x = rotation[0]
    transform.transform.rotation.y = rotation[1]
    transform.transform.rotation.z = rotation[2]
    transform.transform.rotation.w = rotation[3]

    # Broadcast the transform
    static_broadcaster.sendTransform(transform)

if __name__ == '__main__':
    rospy.init_node('tf_broadcaster')

    rospy.loginfo("Dynamic TF broadcaster node started")

    # Initialize global variables
    surface_normal_msg = None
    center_of_pc_msg = None
    static_broadcaster = tf2_ros.StaticTransformBroadcaster()

    global Tws
    # Define the predefined Tws matrix (example)
    Tws = np.eye(4)  # Start with the identity matrix
    # Tws[0:3, 0:3] = np.array([[0.866, -0.5, 0.0],
    #                           [0.5, 0.866, 0.0],
    #                           [0.0, 0.0, 1.0]])  # Example rotation

    translation = rospy.get_param("translation", 0.0)

    Tws[0:3, 3] = np.array([translation, 0.0, 0.0])  # Example translation

    # Make Tws global

    sn_sub = message_filters.Subscriber('/surface_normal', Vector3Stamped)
    copc_sub = message_filters.Subscriber('/center_of_pc', Vector3Stamped)

    ts = message_filters.ApproximateTimeSynchronizer([sn_sub, copc_sub], 10, 0.1)
    ts.registerCallback(calculate_callback)

    rospy.spin()
