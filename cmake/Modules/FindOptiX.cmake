# FindOptiX.cmake
# CMake module to locate OptiX

message(STATUS "Searching for OptiX in ${OptiX_ROOT_DIR}")

find_path(OptiX_INCLUDE_DIR optix.h
          HINTS
          $ENV{OptiX_ROOT_DIR}/include
          PATHS
          /home/lucfra/NVIDIA-OptiX-SDK-8.0.0/include
          DOC "Path to the OptiX include directory")
message(STATUS "OptiX include directory: ${OptiX_INCLUDE_DIR}")


message(STATUS "Searching for OptiX library")
message(STATUS "LD_LIBRARY_PATH: $ENV{LD_LIBRARY_PATH}")
message(STATUS "OptiX_ROOT_DIR: $ENV{OptiX_ROOT_DIR}")
find_library(OptiX_LIBRARY
             NAMES optix
             PATHS
             /home/lucfra/optix_build
             /home/lucfra/optix_build/lib
             DOC "Path to the OptiX library")
message(STATUS "OptiX library: ${OptiX_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX DEFAULT_MSG OptiX_INCLUDE_DIR OptiX_LIBRARY)

if (OptiX_FOUND)
    set(OptiX_INCLUDE_DIRS ${OptiX_INCLUDE_DIR})
    set(OptiX_LIBRARIES ${OptiX_LIBRARY})
else()
    set(OptiX_INCLUDE_DIRS)
    set(OptiX_LIBRARIES)
    message(WARNING "Could NOT find OptiX: missing OptiX_INCLUDE_DIR or OptiX_LIBRARY")
endif()
