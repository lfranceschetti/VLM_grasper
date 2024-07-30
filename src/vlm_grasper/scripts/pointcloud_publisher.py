#!/home/lucfra/miniconda3/envs/ros_env/bin/python
import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2, CameraInfo
from geometry_msgs.msg import TransformStamped
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os
import rospkg
from vlm_grasper.simulator.simulation_clutter_bandit import ClutterRemovalSim
from vlm_grasper.simulator.transform import Rotation, Transform
from vlm_grasper.simulator.io_smi import *
from std_msgs.msg import Header


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.25])
    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    for i in range(n):
        r = np.random.uniform(2, 2.5) * sim.size
        theta = np.random.uniform(np.pi / 4, np.pi / 3)
        phi = np.random.uniform(0.0, np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)

        depth_img = sim.camera.render(extrinsic)[1]
        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        eye = np.r_[
            r * sin(theta) * cos(phi),
            r * sin(theta) * sin(phi),
            r * cos(theta),
        ]
        eye = eye + origin.translation
    return depth_imgs, extrinsics, eye


def convert_intrinsic_to_camera_info_msg(intrinsic, timestamp):
    msg = CameraInfo()
    msg.header.stamp = timestamp
    msg.width = intrinsic.width
    msg.height = intrinsic.height
    msg.K = [intrinsic.fx, 0.0, intrinsic.cx,
             0.0, intrinsic.fy, intrinsic.cy,
             0.0, 0.0, 1.0]
    msg.D = [0, 0, 0, 0, 0]  # Assuming no distortion
    msg.P = [intrinsic.fx, 0.0, intrinsic.cx, 0.0,
             0.0, intrinsic.fy, intrinsic.cy, 0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg


def convert_extrinsics_to_transform_msg(extrinsic, frame_id, child_frame_id, timestamp):
    t = TransformStamped()
    t.header.stamp = timestamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id

    # Assuming the extrinsic array is in the format [x, y, z, qx, qy, qz, qw]
    t.transform.translation.x = extrinsic[0]
    t.transform.translation.y = extrinsic[1]
    t.transform.translation.z = extrinsic[2]

    t.transform.rotation.x = extrinsic[3]
    t.transform.rotation.y = extrinsic[4]
    t.transform.rotation.z = extrinsic[5]
    t.transform.rotation.w = extrinsic[6]

    return t


def get_point_cloud_msg(tsdf, sim, timestamp):
    pc = tsdf.get_cloud()
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    pc = pc.crop(bounding_box)
    header = Header()
    header.stamp = timestamp
    header.frame_id = "camera_link"
    return pc2.create_cloud_xyz32(header, np.asarray(pc.points))




def publisher_node(use_simulation=True):
    rospy.init_node('camera_data_publisher')

    pc_pub = rospy.Publisher('/input_point_cloud', PointCloud2, queue_size=10)
    intrinsics_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=10)
    extrinsics_pub = rospy.Publisher('/camera/extrinsics/depth_to_color', TransformStamped, queue_size=10)

    rate = rospy.Rate(0.1)  # 0.1 Hz = 10 seconds

    # Get the path to the package
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('vlm_grasper')  # Replace 'your_package_name' with your actual package name

    # Choose one Object from src/vlm_grasper/data_robot/egad_eval_set
    if not use_simulation:
        pc_A6 = o3d.io.read_triangle_mesh(os.path.join(package_path, 'src/vlm_grasper/data_robot/egad_eval_set/A6.obj'),
                                          enable_post_processing=True)
        pc_A6.compute_vertex_normals()
        points_A6 = np.asarray(pc_A6.vertices)

    sim = ClutterRemovalSim("obj", "packed/test", gui=False, rand=True)

    while not rospy.is_shutdown():
        if use_simulation:
            sim.reset(1)
            depth_imgs, extrinsics, eye = render_images(sim, 1)


            # Reconstruct point cloud using a subset of the images
            tsdf = create_tsdf(sim.size, 180, depth_imgs, sim.camera.intrinsic, extrinsics)


            # Use a consistent timestamp for all messages
            timestamp = rospy.Time.now()


            pc_msg = get_point_cloud_msg(tsdf, sim, timestamp)

            transform_msg = convert_extrinsics_to_transform_msg(extrinsics[0], "camera_link", "camera_depth_link", timestamp)

            intrinsics_msg = convert_intrinsic_to_camera_info_msg(sim.camera.intrinsic, timestamp)

            # Publish intrinsic parameters
            intrinsics_pub.publish(intrinsics_msg)

            pc_pub.publish(pc_msg)

            extrinsics_pub.publish(transform_msg)


        else:
            pc_pub.publish(pc2.create_cloud_xyz32(Header(frame_id="A6"), points_A6))

        rate.sleep()


if __name__ == '__main__':
    try:
        publisher_node(use_simulation=True)
    except rospy.ROSInterruptException:
        pass
