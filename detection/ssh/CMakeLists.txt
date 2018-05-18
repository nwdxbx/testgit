cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
SET(CMAKE_BUILD_TYPE DEBUG)
project(SSH_detector)
find_package(CUDA)
set(CUDA_HOST_COMPILER g++)
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_61")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "Xcompiler")
list(APPEND CUDA_NVCC_FLAGS "-fPIC")

INCLUDE_DIRECTORIES("/usr/local/cuda/include")
INCLUDE_DIRECTORIES("/usr/local/include")
INCLUDE_DIRECTORIES("/media/e/FrameWork/SSH/caffe-ssh/include")
INCLUDE_DIRECTORIES("/media/e/FrameWork/SSH/caffe-ssh/build/src")


INCLUDE_DIRECTORIES("/usr/include/python2.7")
INCLUDE_DIRECTORIES("/media/e/FrameWork/SSH")
LINK_DIRECTORIES("/usr/lib/python2.7")

LINK_DIRECTORIES("/usr/local/cuda/lib64")
LINK_DIRECTORIES("/usr/local/lib")
LINK_DIRECTORIES("/usr/lib/x86_64-linux-gnu")
LINK_DIRECTORIES("/media/e/FrameWork/SSH/caffe-ssh/build/lib")
add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(SSH_detector opencv_core opencv_imgproc opencv_imgcodecs
                      opencv_highgui opencv_video opencv_videoio cudart cublas caffe
                      glog boost_system)
