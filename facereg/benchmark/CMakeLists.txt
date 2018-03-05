cmake_minimum_required(VERSION 2.6)
add_compile_options(-std=c++11)
SET(CMAKE_BUILD_TYPE DEBUG)
project(main)
find_package(CUDA)
set(CUDA_HOST_COMPILER g++)
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_61")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "Xcompiler")
list(APPEND CUDA_NVCC_FLAGS "-fPIC")

include_directories(${CMAKE_SOURCE_DIR})
#include_directories(/opt/OpenBLAS/include)
include_directories(/usr/local/cuda/include)
INCLUDE_DIRECTORIES(/usr/local/include)
include_directories(/media/e/FrameWork/caffe-ssd/include)
include_directories(/media/e/FrameWork/caffe-ssd/build/src)

LINK_DIRECTORIES(/usr/local/lib)
LINK_DIRECTORIES(/usr/lib/x86_64-linux-gnu)
link_directories(/media/e/FrameWork/caffe-ssd/build/lib)
link_directories(/usr/local/cuda/lib64)


aux_source_directory(${CMAKE_SOURCE_DIR} src_code)
add_executable(main ${src_code} "face_reid_caffe.hpp" "simple_caffe.hpp")

target_link_libraries(main caffe opencv_core
opencv_highgui opencv_imgproc opencv_imgcodecs
opencv_video glog boost_system protobuf)
