
import numpy as np
import rospy

import tf.transformations as tf_trans
import tf2_ros
from geometry_msgs.msg import Pose, Vector3


LENGTH = 0.09

def transform_matrix(input_matrix, from_frame, to_frame):
    # Initialize the tf2 transform listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # Wait for the transform to become available
    try:
        transform = tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0))
    except tf2_ros.LookupException as e:
        rospy.logerr("Transform lookup failed: %s" % str(e))
        return None
    
    # Extract translation and rotation (quaternion) from the transform
    translation = [transform.transform.translation.x,
                   transform.transform.translation.y,
                   transform.transform.translation.z]
    
    rotation = [transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w]

    # Convert quaternion to a 4x4 transformation matrix
    T_camera_world = tf_trans.quaternion_matrix(rotation)
    T_camera_world[0:3, 3] = translation

    # Multiply the input matrix (T_grasp_camera) by the camera-to-world transformation matrix
    T_grasp_world = np.dot(T_camera_world, input_matrix)

    return T_grasp_world

def extract_pose_from_matrix(matrix):
    """Extracts a Pose from a 4x4 transformation matrix."""
    pose = Pose()

    # Extract translation
    pose.position.x = matrix[0, 3]
    pose.position.y = matrix[1, 3]
    pose.position.z = matrix[2, 3]

    # Extract rotation (convert rotation matrix to quaternion)
    quaternion = tf_trans.quaternion_from_matrix(matrix)
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose


def create_transformation_matrix(position, approach, normal):
    # Convert attributes to arrays
    position = np.array([position.x, position.y, position.z])
    approach = np.array([approach.x, approach.y, approach.z])
    normal = np.array([normal.x, normal.y, normal.z])


    z_axis = -approach
    #Make that the normal direction is flipped and shows in the other direction
    x_axis = -normal
    y_axis = np.cross(x_axis, z_axis)  # Compute the missing Y-axis

    x_axis = np.cross(y_axis, z_axis)  # Compute the missing X-axis

    # Normalize the axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # # Create rotation matrix (For some reason, the axes are flipped)
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Create translation vector 
    translation_vector = position
    # We need to move the gripper back by the length of the gripper
    translation_vector +=  1.5 * LENGTH * approach
    

    # Combine into a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix    
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix
