Project(Demo)
add_compile_options(-std=c++11)

#CFLAGS=-g -Wall -Wextra $(pkg-config --cflags gtk+-3.0)
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/boost_1_59_0)
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/freetype/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/freetype/lib)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/boost_1_59_0/stage/lib)
#link_directories(${CMAKE_SOURCE_DIR}/3rdParty/protobuf/lib)
add_executable(analysis_demo analysis_demo.cpp PutChineseToMat.cpp CvxText.cpp ../include/tuzhenalginterface.hpp)
add_executable(feature_cal_demo feature_cal_demo.cpp ../include/tuzhenfeature.h)
add_executable(face_reid_accuracy_test face_reid_accuracy_test.cpp)
add_executable(load_model_test load_model_test.cpp)
add_executable(keypoints_draw keypoints_draw.cpp)
add_executable(keypoints_data_collector keypoints_data_collector.cpp)
add_executable(face_attr_unit_test face_attr_unit_test.cpp)
add_executable(face_landmark_predict_test face_landmark_predict_test.cpp)
add_executable(face_reid_pair_test face_reid_pair_test.cpp)
add_executable(remove_same_video rsv_demo.cpp)

link_directories(${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(analysis_demo freetype tuzhenalg boost_system boost_thread opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_ml opencv_video opencv_videoio opencv_cudaobjdetect opencv_cudaimgproc opencv_cudawarping cairo)
target_link_libraries(feature_cal_demo protobuf tuzhenalg boost_system boost_thread opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_ml opencv_video opencv_videoio opencv_cudaobjdetect opencv_cudaimgproc opencv_cudawarping glog)
target_link_libraries(face_reid_accuracy_test tuzhenalg boost_system boost_thread glog opencv_core opencv_imgproc opencv_imgcodecs  opencv_highgui opencv_ml opencv_video opencv_videoio)
target_link_libraries(load_model_test tuzhenalg boost_system boost_thread caffe opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_ml opencv_video opencv_videoio opencv_cudaobjdetect opencv_cudaimgproc opencv_cudawarping glog)
target_link_libraries(keypoints_draw openpose caffe opencv_core opencv_imgcodecs opencv_highgui)
target_link_libraries(keypoints_data_collector openpose caffe opencv_core opencv_imgcodecs opencv_highgui boost_thread)
target_link_libraries(face_attr_unit_test tuzhenalg openpose caffe opencv_core opencv_imgcodecs opencv_highgui boost_thread)
target_link_libraries(face_landmark_predict_test tuzhenalg openpose caffe opencv_core opencv_imgcodecs opencv_highgui boost_thread)
target_link_libraries(face_reid_pair_test tuzhenalg openpose caffe opencv_core opencv_imgcodecs opencv_highgui boost_thread)
target_link_libraries(remove_same_video tuzhenalg opencv_core opencv_imgcodecs)

set(ANALYSIS_DEMO_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/analysis_demo)
install(
        TARGETS analysis_demo
        RUNTIME DESTINATION bin )

install(
        TARGETS feature_cal_demo
        RUNTIME DESTINATION bin )

install ( FILES analysis_demo.cpp DESTINATION samples )
install ( FILES feature_cal_demo.cpp DESTINATION samples )

CONFIGURE_FILE(
        "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/dependencies.cmake"
        @ONLY
)

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DNDEBUG)
endif()

INSTALL(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/dependencies.cmake")

install ( FILES ${CMAKE_SOURCE_DIR}/samples/CMakeLists.txt.in DESTINATION samples RENAME "CMakeLists.txt")
