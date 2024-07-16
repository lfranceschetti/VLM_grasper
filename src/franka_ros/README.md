# ROS integration for Franka Emika research robots


## How to run the Carteisan Impedance Controller:
1. Launch the gazebo simulation:
```bash
roslaunch franka_gazebo panda.launch  
```
2. Check the current pose of the end effector:
```bash
rostopic echo /cartesian_impedance_example_controller/measured_pose
```

3. Publish the desired pose of the end effector:
```bash
rostopic pub /cartesian_impedance_example_controller/equilibrium_pose geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.5
    y: 0.0
    z: 0.4
  orientation:
    x: -1.0
    y: 0.0
    z: 0.0
    w: 0.0" 
```

## Control the gripper:
1. Enable the gripper controller:
   1. Run `rosrun rqt_controller_manager rqt_controller_manager` to open the controller manager.
   2. Enable the `franka_gripper` controller by right-clicking on it and selecting `Load and Start`.

2. Close the gripper: 
```bash
rostopic pub /franka_gripper/grasp/goal franka_gripper/GraspActionGoal "header: 
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  width: 0.0
  epsilon:
    inner: 0.0
    outer: 0.0
  speed: 10.0
  force: 20.0"
```
3. Open the gripper:
```bash
rostopic pub /franka_gripper/grasp/goal franka_gripper/GraspActionGoal "header: 
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  width: 0.06
  epsilon:
    inner: 0.0
    outer: 0.0
  speed: 10.0
  force: 20.0"
```

## Write / Modify the actual Controller:
Checkout the controller in the files:
- `franka_ros/franka_example_controllers/include/franka_example_controllers/cartesian_impedance_example_controller.h`
- `franka_ros/franka_example_controllers/src/cartesian_impedance_example_controller.cpp`


[![CI](https://github.com/frankaemika/franka_ros/actions/workflows/ci.yml/badge.svg)](https://github.com/frankaemika/franka_ros/actions/workflows/ci.yml)


See the [Franka Control Interface (FCI) documentation][fci-docs] for more information.

## License

All packages of `franka_ros` are licensed under the [Apache 2.0 license][apache-2.0].

[apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0.html
[fci-docs]: https://frankaemika.github.io/docs
