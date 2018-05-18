cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
#SET(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_BUILD_TYPE DEBUG)
project(tzAl)
find_package(CUDA)
set(CUDA_HOST_COMPILER g++)
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_61")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "Xcompiler")
list(APPEND CUDA_NVCC_FLAGS "-fPIC")

INCLUDE_DIRECTORIES("/usr/local/include/opencv")
INCLUDE_DIRECTORIES("/usr/local/include/opencv2")
INCLUDE_DIRECTORIES("/usr/local/include")
INCLUDE_DIRECTORIES("/usr/local/cuda/include")
INCLUDE_DIRECTORIES("/media/e/FrameWork/caffe/include")
INCLUDE_DIRECTORIES("/media/e/FrameWork/caffe/build/src")

LINK_DIRECTORIES("/media/e/FrameWork/caffe/build/lib")
LINK_DIRECTORIES("/usr/local/lib")
LINK_DIRECTORIES("/usr/local/cuda/lib64")
LINK_DIRECTORIES("/usr/lib/x86_64-linux-gnu")

add_executable(${PROJECT_NAME} "main.cpp" "facedetermine.hpp" "mtcnn.hpp" "mtcnn.cpp")

target_link_libraries(tzAl opencv_core opencv_imgproc opencv_imgcodecs
                      opencv_highgui opencv_video opencv_videoio caffe glog boost_system)
