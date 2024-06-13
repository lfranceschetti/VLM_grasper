#!/bin/bash

# Ensure no Conda environment is active
# conda deactivate

# Unset Conda-specific environment variables
unset PYTHONPATH
unset PYTHONHOME

unset PATH

# Reset PATH to prioritize system Python
export PATH=/usr/bin:$PATH

# Verify Python version
echo "Using Python version:"
python3 --version

which python3
which python

# Source ROS setup
