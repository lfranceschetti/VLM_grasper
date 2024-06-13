conda create -n ros_env python=3.10

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install pyg -c pyg
conda install matplotlib
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install rospkg
pip3 install empy==3.3.4
pip install open3d
pip install pybullet trimesh
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
conda install libffi==3.3
