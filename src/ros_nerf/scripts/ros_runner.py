#!/usr/bin/env python3
"""
ros_runner.py
"""

import functools
import logging
import pathlib
import random
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock

import yaml
import actionlib
import cv2
import numpy as np
import PIL.Image as pil
import ros_numpy
import rospy
import tf2_ros
import torch
import tyro
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import \
    FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import \
    ParallelDataManager
from nerfstudio.configs.method_configs import all_methods, all_descriptions
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackLocation)
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from rich.logging import RichHandler
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeRemainingColumn)
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, UInt16
from typing_extensions import Annotated
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from std_srvs.srv import Empty, EmptyResponse, SetBool, SetBoolResponse



try:
    from nerf_teleoperation_msgs.msg import NerfRenderRequestAction as Action
    from nerf_teleoperation_msgs.msg import \
        NerfRenderRequestFeedback as Feedback
    from nerf_teleoperation_msgs.msg import NerfRenderRequestGoal as Goal
    from nerf_teleoperation_msgs.msg import NerfRenderRequestResult as Result
except ImportError as e:
    print("Failed to import nerf_teleoperation, did you source the catkin workspace?")
    sys.exit(1)


import threading

from ros_nerf.data.ros_dataloader import ROSDataloader, ROSEvalDataLoader
from ros_nerf.data.ros_dataparser import ROSDataParserConfig
from ros_nerf.data.ros_dataset import ROSDataset
from ros_nerf.utils.ros_config import ROSConfig

NerfstudioConfig = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions, prefix_names=False)
    ]
]


class Modes(Enum):
    """Modes for rendering output."""

    DYNAMIC = 0
    """Dynamic mode, where the data is rendered only on request."""

    CONTINUOUS = 1
    """Continuous mode, where the data is streamed continuously."""

    FOVEATED = 2
    """WIP _Foveated mode, where the data is streamed continuously._"""

    VR = 3
    """Renders equirectangular images for VR streaming."""


@dataclass
class RunnerConfig:
    """Configuration for running the ROS node with NerfStudio parameters."""

    nerf: NerfstudioConfig
    """NerfStudio method configuration."""

    ros: Annotated[ROSConfig, tyro.conf.arg(name='')] 
    """ROS node configuration settings."""


@dataclass
class RosRunner:
    """Class to maintain the ros node and system."""

    render_resolutions: tuple = (0.4, 1.0)
    """List of percentages to render the images at."""

    _feedback: Feedback = Feedback()
    """Feedback object for ROS action server."""

    _result: Result = Result()
    """Result object for ROS action server."""

    def main(self, runner_config: RunnerConfig) -> None:

        signal.signal(signal.SIGINT, self.sigint_handler)

        with CONSOLE.status("[bold green] Starting ROS node..."):
            rospy.init_node("nerf_studio_server", anonymous=True)

            logging.getLogger('rosout').handlers[0] = RichHandler(
                console=CONSOLE)

        rospy.loginfo("ROS node started successfully!")

        # Create action server for handling render requests
        self._as = actionlib.SimpleActionServer(
            "render_nerf", Action, self.execute, False)
        self._as.start()

        rospy.Service("save_transforms", Empty, self.write_json)


        self.training = False
        self.train_lock = Lock()

        self.mode = Modes.DYNAMIC

        self.buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buffer)

        # Create publishers for step and loss data during training
        self.step_pub = rospy.Publisher(
            "nerf_step", UInt16, queue_size=3, latch=True)
        self.loss_pub = rospy.Publisher(
            "nerf_loss", Float32, queue_size=3, latch=True)

        config = runner_config.nerf
        self.config = config
        ros_config = runner_config.ros

        name = ros_config.update()
        self.name = name


        # Swap out standard datamanager for online ROS one
        # TODO: This is really hacky, but it works for now
        datamanager = None
        if hasattr(config.pipeline.datamanager._target, "__origin__") or config.pipeline.datamanager._target == ParallelDataManager:
            # Currently doesnt support the parallel datamanager so we swap it to a vanilla one

            config.pipeline.datamanager._target = VanillaDataManager[ROSDataset]
            datamanager = 'vanilla'
        else:                   
            config.pipeline.datamanager._target = FullImageDatamanager[ROSDataset]
            datamanager = 'full'

        config.pipeline.datamanager.dataparser = ROSDataParserConfig()

        # Pass the setup ROS config to the dataparser for linking
        config.pipeline.datamanager.dataparser.config = ros_config



        # Create callback to run ever 2 training iterations
        trainer_cb = TrainingCallback(where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                                      func=self.training_cb,
                                      update_every_num_iters=2)

        # Display config settings
        ros_config.print_to_terminal()
        config.experiment_name = name
        config.set_timestamp()
        config.print_to_terminal()
        config.save_config()
        rospy.loginfo(f"Loading config {name}")

        # Publish the config to ROS
        # starting with the model name to the parameter server
        rospy.set_param("~/model_name", config.method_name)
        rospy.set_param("~/config_name", name)

        # TODO: remove when this is fixed in current nerfstudio
        # for now create the folder if it doesn't exist
        exp_path = Path('outputs') / name / \
            config.method_name / config.timestamp
        exp_path.mkdir(parents=True, exist_ok=True)

        try:
            self.set_seed(config.machine.seed)

            # Create trainer using setup factory
            self.trainer = config.setup(local_rank=0, world_size=1)
            self.trainer.setup()

            self.train_dataset = self.trainer.pipeline.datamanager.train_dataset
            self.eval_dataset = self.trainer.pipeline.datamanager.eval_dataset


            self.trainer.pipeline.datamanager.setup_train()
            self.trainer.pipeline.datamanager.setup_eval()


            # Handle overwriting for different datamanagers
            # Full datamanager, we are already caching the images in our dataset so we can pass that directly to the trainer
            if datamanager == 'full':
                self.train_dataset.ray_cameras = self.train_dataset.cameras
                self.eval_dataset.ray_cameras = self.eval_dataset.cameras

                self.trainer.pipeline.datamanager.cached_train = self.train_dataset
                self.trainer.pipeline.datamanager.cached_eval = self.eval_dataset
                
                FullImageDatamanager.next_eval_image = VanillaDataManager.next_eval_image
                self.train_dataset.reset_cameras()


            # If we are using the vanilla datamanager, we need to replace the dataloader with our ROS one
            if datamanager == 'vanilla':      
                dataloader = ROSDataloader(
                    dataset=self.train_dataset,
                    device=self.trainer.pipeline.model.device)

                self.trainer.pipeline.datamanager.train_image_dataloader = dataloader
                
                self.trainer.pipeline.datamanager.iter_train_image_dataloader = iter(dataloader)
                self.train_dataset.ray_cameras = self.trainer.pipeline.datamanager.train_ray_generator.cameras

                eval_image_dataloader = ROSDataloader(
                    dataset=self.eval_dataset,
                    device=self.trainer.pipeline.model.device,
                    )               
                
                self.eval_dataset.ray_cameras = self.trainer.pipeline.datamanager.eval_ray_generator.cameras     
                self.trainer.pipeline.datamanager.eval_image_dataloader = eval_image_dataloader
                self.trainer.pipeline.datamanager.iter_eval_image_dataloader = iter(eval_image_dataloader)



            eval_dataloader = ROSEvalDataLoader(
                input_dataset=self.eval_dataset,
                device=self.trainer.pipeline.model.device,
                idx=ros_config.eval_indicies)

            self.trainer.pipeline.datamanager.eval_dataloader = eval_dataloader
            

            self.base_frame = self.train_dataset.base_frame

            # Register publishing callback
            self.trainer.callbacks.append(trainer_cb)

            # Enter loop for some time waiting for data to be received
            start_time = rospy.get_time()
            timeout_sec = 60    
            num_imgs_start = self.train_dataset.num_start
            status = False

            with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TimeRemainingColumn(), MofNCompleteColumn(),TextColumn("Images loaded"), console=CONSOLE) as progress:
                capture_task = progress.add_task("Capturing initial data...", total=num_imgs_start)

                while not rospy.is_shutdown() and rospy.get_time() - start_time < timeout_sec:

                    current_num_imgs = self.train_dataset.current_idx
                    progress.update(capture_task, completed=current_num_imgs)
                    if current_num_imgs >= num_imgs_start:
                        status = True
                        break
                    time.sleep(0.05)


            if not status:
                rospy.logerr("No data received, shutting down")
                self.sigint_handler()
            else:
                rospy.loginfo("Data received, starting training")
                self.training = True

                with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TimeRemainingColumn(), MofNCompleteColumn(),TextColumn("Images loaded"), console=CONSOLE) as progress:
                    capture_task = progress.add_task("Capturing data...", total=self.train_dataset.num_images)
                    # Starts the main nerfstudio training loop
                    def start_training():
                        self.trainer.train()

                    # Start training in a separate thread
                    self.training_thread = threading.Thread(target=start_training)
                    self.training_thread.daemon = True
                    self.training_thread.start()
                    while not rospy.is_shutdown() and self.training_thread.is_alive():
                        current_num_imgs = self.train_dataset.current_idx
                        progress.update(capture_task, completed=current_num_imgs)
                        if current_num_imgs >= self.train_dataset.num_images:
                            break
                        time.sleep(0.05)

            self.training_thread.join()
            
            self.train_dataset.run = False
            self.eval_dataset.run = False

            rospy.loginfo("Training finished")

        except Exception as e:
            rospy.logerr(e)
            traceback.print_exc()
            self.sigint_handler()
        finally:
            profiler.flush_profiler(config.logging)

    def training_cb(self, step: int) -> None:
        """Callback to run before each training iteration."""

        # Publish training step
        step_msg = UInt16()
        step_msg.data = step
        self.step_pub.publish(step_msg)

        # Publish training loss
        loss_msg = Float32()
        # with torch.no_grad():
        with torch.autocast(device_type='cpu'):

            # Get loss dict from the model
            _, loss_dict, metrics_dict = self.trainer.pipeline.get_train_loss_dict(
                step=step)

            # Sum all losses
            loss = functools.reduce(
                torch.add, loss_dict.values()).detach().cpu().numpy()

            # Unstable training, such as with incorrect intrinsics, can cause NaN loss and an all black render
            if np.isnan(loss):
                rospy.logerr("Loss is nan!!")
                loss = 0
                # self.sigint_handler()

            loss_msg.data = loss
            self.loss_pub.publish(loss_msg)

    def execute(self, goal: Goal) -> None:
        """Callback for ROS action server."""

        rospy.loginfo("Received render goal")

        # Check that training has started and is not paused
        if self.training and (self.trainer.training_state == "training" or self.trainer.training_state == "completed"):

            stime = time.time()

            # Convert the camera pose to transform matrix
            pose_msg = goal.pose
            pose = ros_numpy.numpify(pose_msg)

            # Get the transform from the base frame to the camera frame
            tf = self.buffer.lookup_transform(
                self.base_frame, goal.frame_id, rospy.Time(0), rospy.Duration(1.0))
            tf = ros_numpy.numpify(tf.transform)

            base_pose = tf@pose

            # Create a compressed image message for sending render
            compressed_msg = CompressedImage()
            compressed_msg.format = "jpeg"

            client_id = goal.client_id

            # Convert the pose to a tensor for NerfStudio
            c2w = torch.tensor(base_pose[:3, :])
            c2w = c2w.to(dtype=torch.float32)

            dtime = time.time() - stime
            rospy.logdebug(f"Time to get pose: {dtime}", style="bold green")

            img_ = np.zeros((goal.height, goal.width, 4), dtype=np.uint8)
            self.mode = Modes(goal.mode)

            # Get training lock to prevent too many resources being used, TODO: verify with threaded setup
            with self.trainer.train_lock:

                stime = time.time()

                # Set the render bounding box if included in the goal
                if goal.box_size > 0:
                    rospy.loginfo("Cropping to box size: %s" % goal.box_size)

                    crop_min = torch.tensor(
                        [-1000, -1000, -goal.box_size], dtype=torch.float32)
                    crop_max = torch.tensor(
                        [1000, 1000, goal.box_size], dtype=torch.float32)

                    # Either update or create the render bounding box
                    if isinstance(self.trainer.pipeline.model.render_aabb, SceneBox):
                        self.trainer.pipeline.model.render_aabb.aabb[0] = crop_min
                        self.trainer.pipeline.model.render_aabb.aabb[1] = crop_max
                    else:
                        self.trainer.pipeline.model.render_aabb = SceneBox(
                            aabb=torch.stack([crop_min, crop_max], dim=0))
                else:
                    self.trainer.pipeline.model.render_aabb = None

                # Begin rendering at each resolution
                for i, res in enumerate(self.render_resolutions):
                    t = time.time()

                    res *= goal.resolution

                    # Scaling for foveated rendering
                    if self.mode == Modes.FOVEATED:
                        scale = 1 - (i/len(self.render_resolutions))
                    else:
                        scale = 1

                    height = int(goal.height*res*scale)
                    width = int(goal.width*res*scale)

                    rospy.loginfo(
                        f"Render at {width}x{height}, scale {scale} res {res}")

                    # Setup the cameras for rendering based on perspective or VR rendering mode
                    fl = (height/2) / goal.fov_factor
                    if self.mode == Modes.VR:  # equirectangular
                        pph = height / 2
                        ppw = height
                        fx = float(ppw)
                        cx = fx
                        fy = float(pph)
                        cy = fy
                        fy *= 2
                        camType = CameraType.EQUIRECTANGULAR
                        w = int(ppw*2)
                        h = int(pph*2)
                    else:
                        pph = height
                        ppw = width
                        fx = fl
                        cx = ppw / 2
                        fy = fl
                        cy = pph / 2
                        camType = CameraType.PERSPECTIVE
                        w = int(ppw)
                        h = int(pph)

                    camera = Cameras(fx=fx,
                                     fy=fy,
                                     cx=cx,
                                     cy=cy,
                                     width=w,
                                     height=h,
                                     camera_type=camType,
                                     camera_to_worlds=c2w[None, ...],
                                     times=torch.tensor([0.0]),)

                    # Set the camera to eval mode and move to GPU
                    camera = camera.to(
                        device=self.trainer.pipeline.model.device)
                    self.trainer.pipeline.model.eval()

                    # Generate the camera ray bundle and render the image
                    with torch.no_grad():
                        # if self.name == "splatfacto":
                            # camera_ray_bundle = camera
                        # else:
                            # camera_ray_bundle = camera.generate_rays(
                                # camera_indices=0, aabb_box=self.trainer.pipeline.model.render_aabb)

                        outputs = self.trainer.pipeline.model.get_outputs_for_camera(
                            camera)

                        etime = time.time()
                        rospy.loginfo(f"Rendered in {etime - t} at {res}%")

                        try:
                            # Some models don't have depth, or its named differently, but prefer expected_depth
                            try: 
                                depth_img_ = outputs['expected_depth'].cpu().numpy()
                            except KeyError:
                                depth_img_ = outputs['depth'].cpu().numpy()

                            depth_img_ = cv2.resize(depth_img_, (goal.width, goal.height), interpolation=cv2.INTER_NEAREST)

                        except Exception as e:
                            rospy.logerr(outputs.keys())
                            rospy.logerr(e)

                        if self.mode == Modes.FOVEATED:
                            # For foveated rendering, we need to scale the image and place it in the center of the output

                            tmp = cv2.resize(outputs['rgb'].cpu().numpy()[..., ::-1]*255, (int(
                                goal.width*scale), int(goal.height*scale)), interpolation=cv2.INTER_NEAREST)
                            img_[int(img_.shape[0]*(1-scale))//2:int(img_.shape[0]*(1-scale))//2+tmp.shape[0], int(
                                img_.shape[1]*(1-scale))//2:int(img_.shape[1]*(1-scale))//2+tmp.shape[1], :3] = tmp
                        else:
                            # Otherwise, just resize the image to the output resolution
                            img_[:, :, :3] = cv2.resize(outputs['rgb'].cpu().numpy(
                            )[..., ::-1]*255, (goal.width, goal.height), interpolation=cv2.INTER_NEAREST)

                        rospy.loginfo(
                            f"min max depth {np.min(depth_img_)}  {np.max(depth_img_)}")

                        # Scale the depth image to 0-1 for better support outside of ROS
                        depth_img_ = 1 / (depth_img_ + 1)

                        self._feedback.resolution = res
                        self._feedback.client_id = client_id

                        compressed_msg.data = cv2.imencode('.jpg', img_)[
                            1].tostring()

                        # Send feeback at the lower resolutions
                        self._feedback.rgb_image = compressed_msg
                        self._feedback.depth_image = ros_numpy.msgify(
                            Image, depth_img_, encoding="32FC1")
                        self._as.publish_feedback(self._feedback)

                        CONSOLE.print(
                            f"Published feedback in {time.time() - etime}", style="bold green")

                    # Resume training
                    self.trainer.pipeline.model.train()

                    if self._as.is_preempt_requested():
                        self._as.set_preempted()
                        rospy.loginfo("Render preempted")
                        return

                rospy.logdebug(f"Rendered in {time.time() - stime}")
        else:
            self._as.set_preempted()
            rospy.loginfo("Render not possible, no training data")
            return

        # If we are in continuous mode, keep rendering until preempted
        if self.mode == Modes.CONTINUOUS:
            with torch.no_grad():
                while not self._as.is_preempt_requested() and not rospy.is_shutdown() and self._as.is_active():

                    tf = self.buffer.lookup_transform(
                        self.base_frame, goal.frame_id, rospy.Time(0), rospy.Duration(1.0))
                    tf = ros_numpy.numpify(tf.transform)

                    base_pose = tf@pose

                    camera.camera_to_worlds[0] = torch.tensor(base_pose[:3, :])
                    camera_ray_bundle = camera.generate_rays(
                        camera_indices=0, aabb_box=self.trainer.pipeline.model.render_aabb)
                    outputs = self.trainer.pipeline.model.get_outputs_for_camera_ray_bundle(
                        camera_ray_bundle)


                    try: 
                        depth_img_ = outputs['expected_depth'].cpu().numpy()
                    except KeyError:
                        depth_img_ = outputs['depth'].cpu().numpy()

                    depth_img_ = cv2.resize(
                        depth_img_, (goal.width, goal.height), interpolation=cv2.INTER_NEAREST)

                    img_ = np.zeros(
                        (goal.height, goal.width, 4), dtype=np.uint8)

                    img_[:, :, :3] = cv2.resize(outputs['rgb'].cpu().numpy(
                    )[..., ::-1]*255, (goal.width, goal.height), interpolation=cv2.INTER_NEAREST)
                    img_[:, :, 3] = (np.inf == depth_img_)*250

                    depth_img_[depth_img_ > 10000] = -1

                    rospy.loginfo(
                        f"min max depth {np.min(depth_img_)}  {np.max(depth_img_)}, state {self._as.is_active()}, preempt {self._as.is_preempt_requested()} new goal {self._as.is_new_goal_available()}")

                    depth_img_ = 1 / (depth_img_ + 1)
                    self._feedback.client_id = client_id

                    self._feedback.resolution = res
                    compressed_msg.data = cv2.imencode('.jpg', img_)[
                        1].tostring()

                    # Publish the continuous feedback
                    self._feedback.rgb_image = compressed_msg
                    self._feedback.depth_image = ros_numpy.msgify(
                        Image, depth_img_, encoding="32FC1")
                    self._as.publish_feedback(self._feedback)

            self._as.set_preempted()
            rospy.loginfo("Render preempted")
            return

        compressed_msg.data = cv2.imencode('.jpg', img_)[1].tostring()

        # Publish the final result
        self._result.rgb_image = compressed_msg
        self._result.depth_image = ros_numpy.msgify(
            Image, depth_img_, encoding="32FC1")
        self._result.client_id = client_id
        self._result.render_time = time.time()-t
        self._result.resolution = res

        self._as.set_succeeded(self._result)

    def sigint_handler(self, sig=None, frame=None):
        """Handles SIGINT signal and passes on to rospy."""
        CONSOLE.print("[bold red]SIGINT received, shutting down...")
        rospy.signal_shutdown("Keyboard interrupt")
        sys.exit(0)

    def set_seed(self, seed: int) -> None:
        """Sets the random seed for the experiment."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def write_json(self, req: Empty) -> EmptyResponse:
        """Writes the current train dataset to files"""

        if not self.train_dataset:
            rospy.logerr("No train dataset found")
            return EmptyResponse()
        
        self.train_dataset.save(Path('data') / self.name / self.config.timestamp)
        rospy.loginfo("Saved train dataset")

        return EmptyResponse()


def entrypoint():
    """Entrypoint for running the ros node and system directly from cli."""
    tyro.extras.set_accent_color("#367ac6")
    ros = RosRunner()

    # merge annotated config with rosconfig

    ros.main(tyro.cli(RunnerConfig))


if __name__ == "__main__":
    entrypoint()
