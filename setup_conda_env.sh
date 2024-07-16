pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install matplotlib
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install rospkg
pip3 install empy==3.3.4
pip install open3d==0.16.0
pip install pybullet trimesh
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
conda install libffi==3.3
conda install libtiff==4.4.0
conda install cffi==1.14.0

# # TO run rosbag scripts
conda install -c conda-forge pycryptodomex
conda install -c conda-forge opencv

# pip install gnupg

#For Segment anything
pip install git+https://github.com/facebookresearch/segment-anything.git
#For Segment anything mask annotation
pip install supervision

#Maybe (only do it if needed, it is necessary for some postprocessing, which im not sure if we use)
#pip install opencv-python pycocotools matplotlib onnxruntime onnx


#USE python 3.10







