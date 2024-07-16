#!/bin/bash
rosbag record -e "(.*)/cartesian_impedance_example_controller/(.*)|(.*)/franka_state_controller/F_ext|/particles|/mission_control/planned_path|/tf|/tf_static"