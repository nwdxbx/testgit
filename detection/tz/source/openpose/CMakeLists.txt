cmake_minimum_required(VERSION 2.8)

add_compile_options(-std=c++11)

project(newopenpose)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

find_package(CUDA)
#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61")

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler")
list(APPEND CUDA_NVCC_FLAGS "-fPIC")
#list(APPEND CUDA_NVCC_FLAGS "-ccbin=g++")
#list(APPEND CUDA_NVCC_FLAGS "-c")

include_directories(${CMAKE_SOURCE_DIR}/source/config_src)
#include_directories(${CMAKE_SOURCE_DIR}/../common/caffe-ssd/include)
#link_directories(${CMAKE_SOURCE_DIR}/../common/caffe-ssd/lib)
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/caffe/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/caffe/lib)

#include_directories("/opt/OpenBLAS/include")
#link_directories("/opt/OpenBLAS/lib")


#CUDA_CUDART_LIBRARY(/usr/local/cuda/lib64)

CUDA_COMPILE(CU_O 
openposeNms.cu
openposeResize.cu)

CUDA_ADD_LIBRARY(openpose SHARED
openposeCaffe.cpp
openposeConnector.cpp
property.cpp
${CU_O}
)


target_link_libraries(openpose caffe boost_system glog 
opencv_core
opencv_imgproc opencv_imgcodecs
opencv_video opencv_videoio cudart cublas)

install(
TARGETS openpose
LIBRARY DESTINATION lib )

set(MY_BINARY_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libopenpose.so)

#CONFIGURE_FILE(
#  "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.cmake.in"
#  "${CMAKE_CURRENT_BINARY_DIR}/dependencies.cmake"
#  @ONLY
#  )
#
#INSTALL(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/dependencies.cmake")
