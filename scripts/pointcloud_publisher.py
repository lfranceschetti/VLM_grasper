#!/home/lucfra/miniconda3/envs/ros_env/bin/python
import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2, CameraInfo
from geometry_msgs.msg import Transform as Ts
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os
import rospkg  # Import rospkg to find the package path
from VLM_grasper.simulator.simulation_clutter_bandit import ClutterRemovalSim
from VLM_grasper.simulator.transform import Rotation, Transform
from VLM_grasper.simulator.io_smi import *
from std_msgs.msg import Header



def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.25])
    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    for i in range(n):
        r = np.random.uniform(2, 2.5) * sim.size
        theta = np.random.uniform(np.pi/4, np.pi/3)
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



def convert_intrinsic_to_camera_info_msg(intrinsic):
    msg = CameraInfo()
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

def convert_extrinsics_to_transform_msg(extrinsic):
    msg = Ts()

    # Assuming the extrinsic array is in the format [x, y, z, qx, qy, qz, qw]
    msg.translation.x = extrinsic[0]
    msg.translation.y = extrinsic[1]
    msg.translation.z = extrinsic[2]

    msg.rotation.x = extrinsic[3]
    msg.rotation.y = extrinsic[4]
    msg.rotation.z = extrinsic[5]
    msg.rotation.w = extrinsic[6]

    return msg


def get_point_cloud_msg(tsdf, sim):
    pc = tsdf.get_cloud()
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    pc = pc.crop(bounding_box)
    return pc2.create_cloud_xyz32(Header(frame_id="camera_link"), np.asarray(pc.points))



def publisher_node(use_simulation=True):
    rospy.init_node('pointcloud_publisher')
    pub = rospy.Publisher('/input_point_cloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(0.05)  # 0.1 Hz = 10 seconds

    # Get the path to the package
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('VLM_grasper')  # Replace 'your_package_name' with your actual package name


    #Choose one Object from  src/VLM_grasper/data_robot/egad_eval_set
    if not use_simulation:
        pc_A6 = o3d.io.read_triangle_mesh(os.path.join(package_path, 'src/VLM_grasper/data_robot/egad_eval_set/A6.obj'), enable_post_processing=True)
        pc_A6.compute_vertex_normals()
        points_A6 = np.asarray(pc_A6.vertices)

    sim = ClutterRemovalSim("obj", "packed/test", gui=False, rand=True)

    while not rospy.is_shutdown():


        if use_simulation:
            sim.reset(1)
            depth_imgs, extrinsics, eye = render_images(sim, 1)

    

            # reconstrct point cloud using a subset of the images
            tsdf = create_tsdf(sim.size, 180, depth_imgs, sim.camera.intrinsic, extrinsics)
            pc_msg = get_point_cloud_msg(tsdf, sim)
            
            # Add the point cloud to the message
            pub.publish(pc_msg)
            

        else:
            pub.publish(pc2.create_cloud_xyz32(Header(frame_id="A6"), points_A6))

        rate.sleep()




if __name__ == '__main__':
    try:
        publisher_node(use_simulation=True)
    except rospy.ROSInterruptException:
        pass
