# Install script for directory: /media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data"
         RPATH "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/examples/cifar10/convert_cifar_data")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data"
         OLD_RPATH "/usr/local/cuda/lib64:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib::::::::"
         NEW_RPATH "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_cifar_data")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data"
         RPATH "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/examples/mnist/convert_mnist_data")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data"
         OLD_RPATH "/usr/local/cuda/lib64:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib::::::::"
         NEW_RPATH "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_data")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data"
         RPATH "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/examples/siamese/convert_mnist_siamese_data")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data"
         OLD_RPATH "/usr/local/cuda/lib64:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib::::::::"
         NEW_RPATH "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/convert_mnist_siamese_data")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification"
         RPATH "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/examples/cpp_classification/classification")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification"
         OLD_RPATH "/usr/local/cuda/lib64:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib::::::::"
         NEW_RPATH "/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/project/caffe-ocr/cmake/install/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib:/media/liujin/8f304754-fa23-4d25-bdd9-8095e1958bb7/opencv-3.2.0/cmake/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/classification")
    endif()
  endif()
endif()

