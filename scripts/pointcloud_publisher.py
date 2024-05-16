#!/home/lucfra/miniconda3/envs/ros_env/bin/python
import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2
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




def convert_open3d_to_ros(o3d_cloud, frame_id="world"):
    # Convert Open3D point cloud to a list of 3D points
    xyz_points = np.asarray(o3d_cloud.points, dtype=np.float32)
    # Create a ROS PointCloud2 message
    header = Header(frame_id=frame_id, stamp=rospy.Time.now())
    ros_cloud = pc2.create_cloud_xyz32(header, xyz_points)
    return ros_cloud



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
            pc = tsdf.get_cloud()

            # crop surface and borders from point cloud
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
            # o3d.visualization.draw_geometries([pc])
            pc = pc.crop(bounding_box)

            ros_cloud = convert_open3d_to_ros(pc)


            pub.publish(ros_cloud)
        else:
            pub.publish(pc2.create_cloud_xyz32(Header(frame_id="A6"), points_A6))

        rate.sleep()




if __name__ == '__main__':
    try:
        publisher_node(use_simulation=True)
    except rospy.ROSInterruptException:
        pass
