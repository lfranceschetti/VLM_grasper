# NerfStudio Plugin for ROS Multi-Camera Systems
## Introduction
This package combines NerfStudio with ROS to provide a multi-camera system for data collection and visualization. This includes a ROS node for NerfStudio training, helper scripts to interface with the NerfStudio CLI, as well as an RVIZ plugin for visualization.

## Installation

Start by cloning this repository into your catkin workspace and using [rosdep](http://wiki.ros.org/rosdep) to install system packages. For now install is done twice, first by linking the python environment to NerfStudio. If NerfStudio is in a conda environment, activate that first then go into this folder and run `pip install -e .`.

Next build the package as normal with `catkin build nerf_teleoperation`, then source the workspace. Double check installation worked with `ns-ros -h` this should display the help message along with all the installed NerfStudio methods. Also verify that the RViz plugin works by launch RViz with the workspace sourced and switching to the NerfViewController type in the Views panel.


## Usage
### Configuration
Both the online and offline modes are setup using the configuration files located in the config/ folder. Their parameters and layout are as follows:
```yaml
---
# Base config for the NerfStudio ROS sensors

height: int  # image height
width: int   # image width
fx: float    # focal length in pixels [optional]
fy: float    # focal length in pixels [optional]
cx: float    # principal point in pixels [optional]
cy: float    # principal point in pixels [optional]
k1: float    # radial distortion coefficient [optional]
k2: float    # radial distortion coefficient [optional]
k3: float    # radial distortion coefficient [optional]
p1: float    # tangential distortion coefficient [optional]
p2: float    # tangential distortion coefficient [optional]
base_frame: string   # root frame for the camera system, should be fixed such as "world" or "map"
use_preset_D: bool   # whether to force the preset distortion parameters [optional]
num_images: int      # number of images to capture
num_start: int       # number of images to capture before training starts, defaults to all images [optional]
hz: int              # capture rate in hz
blur_threshold: int  # threshold for blur detection between 0 and 100
cameras:             # array of cameras to subscribe too
	- name: string         # name of the camera
		image_topic: string  # topic for the rgb image
		depth_topic: string  # topic for the depth image [optional]
		info_topic:  string  # topic for the camera info
		camera_frame: string # frame of the camera
```


### Offline Training

```bash
ns-ros-save --config_path <config_file> --save_path <output_folder> --no-run-on-start [OPTIONS]
```

To just capture data in the NerfStudio format for later use with a non-ROS NerfStudio install, run the `ns-ros-save` command. This will start the ROS node, and save the data into an rgb/, and depth/ folder if applicable. Various `[OPTIONS]` arguments may also be specified to overwrite any of the components in the config file shown above. For example `--base-frame odom` or `--num_start 5` to overwrite the parameters from the config. The program will automatically save the data once the preset number of images is reached, or the rosservice is trigged with `rosservice call /save_transforms`, or the program is terminated with `ctrl-c`. The capturing can be paused and resumed using the `rosservice call /toggle`.


### Online Training
````bash
ns-ros <model-name> --config_path <config_file> [OPTIONS]
````

To train a model with the ROS node, run the `ns-ros` command with some `<model-name>`. Any model installed and working with `ns-train` should work with the ROS node. The `<config_file>` specifies the path to sensor config detailing all the ROS topics to subscribe to and node parameters. Various `[OPTIONS]` arguments may also be specified to overwrite any of the components in the config file shown above. For example `--base-frame odom` or `--num_start 5` to overwrite the parameters from the config. This will start the ROS node, and wait for data to start streaming. It will then begin training the model based on the config file and allow visualization from the NerfStudio viewer, or interaction via the action server. 

### Visualization
With the workspace sourced, the RVIZ plugin should be linked, allowing you to select "NerfViewController" from the Type dropdown in the Views panel. This will send render requests to the NerfStudio node, and display the renders in the RVIZ window.




## Troubleshooting
If there is an issue with the conda environment and the catkin_ws, you can make catkin use the conda environment for python path by cleaning and rebuilding the workspace with the conda environment activated.


## TODOs
- [x] Support for ns-cli start or rosrun/launch
- [x] ~~Update environment file~~ replaced with pyproject toml install specs
- [x] Launch/run args supported via tyro
- [x] NS package format
- [x] Clean up ros package format
- [x] Viewer args ....?
- [x] Update RIZ plugin
    - [x] Update RIZ plugin for new client model
    - [x] Fix Namespace errors in RIZ plugin
    - [x] Better resize handling
- [ ] New Action server (multi client)
- [ ] Better threading for action server
- [ ] Add vcstool/rosinstall support
- [x] Config from arg
- [x] readme instructions
- [x] reorder into datamanager/parser only
- [ ] support for depth mixed cameras
- [ ] support for lidar depth supervision
- [ ] support for better mixed res sources? **
- [ ] add ros support as custom viewer (common tcp)
    - [ ] add raw splat data over tcp
    - [ ] deformation field/dynamic data
- [ ] multi model support within ros node?
- [ ] better compression
- [x] better caching
- [ ] forgetting mechanism/study
- [x] changing root frame
- [x] setup ros config on tyro for sensors
- [~] verify setup with ns-export/similar command (mostly works, needs to set for inference mode)
- [x] gaussian splatting support
	- [x] densification (later/dynamic splat count/positition)
	- [ ] lidar gaussian init
	- [x] full images manager support
	- [x] ns bump to 1.0.0
	- [x] splat export works if set to inference mode in exporter.py and the assert is disabled for inference mode
	- [ ] splat export service call
- [x] verify depth data support
- [x] ns-ros-save
- [x] plugin - fix depth issue
- [x] yaml config
- [ ] Figure out whats slowing down the splat training.....
- [x] Investigate where cameras are called and 0 focal length breaks training (GPU vs CPU camera instances)
- [ ] Break training console into panel for cleaner printing
- [ ] Save data on service call
- [x] Initial rotation issue
- [x] Initial camera at 0,0,0 issue
- [x] Verify eval works
- [x] Eval fixed indices
	- [ ] Globally fixed indices