cmake_minimum_required(VERSION 3.0.2)
project(VLM_grasper)
## Compile as C++11, supported in ROS Kinetic and newer

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

find_package(CUDA REQUIRED)


## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()


add_message_files(
  FILES
  PointCloudWithGrasps.msg
  Grasp.msg
)



## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################



catkin_package(
 INCLUDE_DIRS include
 LIBRARIES optix_wrapper
 CATKIN_DEPENDS message_runtime rospy std_msgs geometry_msgs sensor_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  /home/lucfra/NVIDIA-OptiX-SDK-8.0.0/include # Path to OptiX include directory
  /usr/local/cuda/include
  ${PROJECT_SOURCE_DIR}/include  # Include your own headers
)


link_directories(
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  /usr/lib/wsl/lib
)



## Declare a C++ library
add_library(optix_wrapper
  src/VLM_grasper/optix/optix_wrapper.cpp
)

## either from message generation or dynamic reconfigure
add_dependencies(optix_wrapper ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


add_executable(${PROJECT_NAME}_node src/VLM_grasper/optix/main.cpp src/VLM_grasper/optix/optix_wrapper.cpp)


set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")


add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})




target_link_libraries(optix_wrapper
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_curand_LIBRARY}
  /usr/lib/wsl/lib/libnvoptix.so.1
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  optix_wrapper
  ${CUDA_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
  cuda  # Link the CUDA driver API
  /usr/lib/wsl/lib/libnvoptix.so.1
)


## Mark executable scripts for installation
catkin_install_python(PROGRAMS
  scripts/edge_grasp_node.py
  scripts/grasp_visualizer.py
  scripts/pointcloud_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


