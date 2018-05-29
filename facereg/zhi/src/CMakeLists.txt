configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h @ONLY )

include_directories(/usr/local/cuda/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/boost_1_59_0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/caffe/include)
#include_directories(/home/zhfyuan/libraries/caffe/build/install/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/boost_1_59_0/stage/lib)



aux_source_directory(. src_code)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/mtcnn src_code_mtcnn)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/compare src_code_compare)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/ZBase64 src_base64)

add_library(face_compare SHARED ${src_code} ${src_code_mtcnn} ${src_code_compare} ${src_base64})

target_link_libraries(face_compare caffe boost_system boost_thread boost_date_time)
set_target_properties(face_compare PROPERTIES VERSION "${VERSION_MAJOR}.${VERSION_MINER}.${VERSION_PATCH}")


install(
TARGETS face_compare
LIBRARY DESTINATION lib )

install ( FILES ../include/zhihui_alg_interface.hpp DESTINATION include )

set(MY_BINARY_LOCATION ${LIBRARY_OUTPUT_PATH}/libface_compare.so)
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/dependencies.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/dependencies.cmake"
  @ONLY
  )

INSTALL(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/dependencies.cmake")
