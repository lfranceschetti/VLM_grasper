#!/bin/bash

# Ensure no Conda environment is active
# conda deactivate

# Unset Conda-specific environment variables
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset CPATH
unset PKG_CONFIG_PATH
unset PYTHONHOME
unset CONDA_PREFIX

# Reset PATH to prioritize system Python
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/ros/noetic/bin:$PATH

# Explicitly set python and python3 to use the system version
export PYTHON=/usr/bin/python3
export PYTHON3=/usr/bin/python3

export CMAKE_PREFIX_PATH=/opt/ros/noetic

# Verify Python version
echo "Using Python version:"
$PYTHON3 --version
which $PYTHON3
which $PYTHON

# Source ROS setup
source /opt/ros/noetic/setup.bash

d ~/catkin_ws
catkin config -DPYTHON_EXECUTABLE=$PYTHON3
catkin clean -y
catkin_make
