cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)

project(zhihui)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv-3.2.0/include)

link_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv-3.2.0/lib)
link_directories(${CMAKE_SOURCE_DIR}/src/include/caffe/build/lib)

link_directories(${CMAKE_SOURCE_DIR}/src/lib)
link_directories("/usr/local/cuda/lib64")

set(VERSION_MAJOR 1)                                                 
set(VERSION_MINER 0)
set(VERSION_PATCH 0)

add_subdirectory(src)

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")
add_executable(zhihui ${CMAKE_SOURCE_DIR}/test/test.cpp ${CMAKE_SOURCE_DIR}/include/zhihui_alg_interface.hpp)
target_link_libraries(zhihui face_compare opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_ml opencv_video opencv_videoio)

install ( FILES ./test/test.cpp DESTINATION .)

