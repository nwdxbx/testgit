project(tuzhenalg)

add_subdirectory(openpose)
#add_subdirectory(track)
#add_subdirectory(report)

configure_file( version.h.in ${CMAKE_SOURCE_DIR}/source/version.h @ONLY )

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/source)
include_directories(${CMAKE_SOURCE_DIR}/source/compare)
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/caffe/include)

aux_source_directory(${CMAKE_SOURCE_DIR}/source src_code)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/track src_code_track)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/compare src_code_compare)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/getstructfeature src_code_getfeature)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/reIDfeature src_code_reIDfeature)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/crowd src_code_crowd)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/mtcnn src_code_mtcnn)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/mask_rcnn src_mask_rcnn)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/vehicle_analysis src_vehicle_analysis)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/pedestrain_analysis src_pedestrain_analysis)
aux_source_directory(${CMAKE_SOURCE_DIR}/source/remove_same_video src_rsv)

#if( CMAKE_BUILD_TYPE STREQUAL "Release" )
#    set(CMAKE_C_FLAGS_DEBUG "-DNDEBUG")
#    set(CMAKE_CXX_FLAGS_DEBUG "-DNDEBUG")
#endif( CMAKE_BUILD_TYPE STREQUAL "Release" )
if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DNDEBUG)
endif()

add_library(tuzhenalg SHARED ${src_code} ${src_mask_rcnn} ${src_vehicle_analysis} ${src_code_track}
    ${src_code_compare} ${src_code_getfeature} ${src_code_reIDfeature} ${src_code_crowd} ${src_code_mtcnn} ${src_pedestrain_analysis} ${src_rsv})
target_link_libraries(tuzhenalg caffe  boost_date_time openpose)

set_target_properties(tuzhenalg PROPERTIES VERSION "${VERSION_MAJOR}.${VERSION_MINER}.${VERSION_PATCH}")


install(
        TARGETS tuzhenalg
        LIBRARY DESTINATION lib )
