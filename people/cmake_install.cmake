# Install script for directory: /home/roitberg/workspace/pcl/gpu/people

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "RelWithDebInfo")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "pcl_gpu_people")
  FOREACH(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpcl_gpu_people.so.1.7.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpcl_gpu_people.so.1.7"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpcl_gpu_people.so"
      )
    IF(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      FILE(RPATH_CHECK
           FILE "${file}"
           RPATH "/usr/local/lib:/usr/lib/openmpi/lib")
    ENDIF()
  ENDFOREACH()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/roitberg/workspace/pcl/lib/libpcl_gpu_people.so.1.7.2"
    "/home/roitberg/workspace/pcl/lib/libpcl_gpu_people.so.1.7"
    "/home/roitberg/workspace/pcl/lib/libpcl_gpu_people.so"
    )
  FOREACH(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpcl_gpu_people.so.1.7.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpcl_gpu_people.so.1.7"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libpcl_gpu_people.so"
      )
    IF(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      FILE(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/roitberg/workspace/pcl/lib:/usr/lib/openmpi/lib:"
           NEW_RPATH "/usr/local/lib:/usr/lib/openmpi/lib")
      IF(CMAKE_INSTALL_DO_STRIP)
        EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "${file}")
      ENDIF(CMAKE_INSTALL_DO_STRIP)
    ENDIF()
  ENDFOREACH()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "pcl_gpu_people")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "pcl_gpu_people")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/roitberg/workspace/pcl/gpu/people/pcl_gpu_people-1.7.pc")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "pcl_gpu_people")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "pcl_gpu_people")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pcl-1.7/pcl/gpu/people" TYPE FILE FILES
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/label_common.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/people_detector.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/face_detector.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/label_segment.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/tree_train.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/label_tree.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/colormap.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/bodyparts_detector.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/label_blob2.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/organized_plane_detector.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/person_attribs.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/probability_processor.h"
    "/home/roitberg/workspace/pcl/gpu/people/include/pcl/gpu/people/tree.h"
    "/home/roitberg/workspace/pcl/gpu/people/src/cuda/nvidia/NCVPixelOperations.hpp"
    "/home/roitberg/workspace/pcl/gpu/people/src/cuda/nvidia/NCVAlg.hpp"
    "/home/roitberg/workspace/pcl/gpu/people/src/cuda/nvidia/NCVRuntimeTemplates.hpp"
    "/home/roitberg/workspace/pcl/gpu/people/src/cuda/nvidia/NCVColorConversion.hpp"
    "/home/roitberg/workspace/pcl/gpu/people/src/cuda/nvidia/NCVPyramid.hpp"
    "/home/roitberg/workspace/pcl/gpu/people/src/cuda/nvidia/NPP_staging.hpp"
    "/home/roitberg/workspace/pcl/gpu/people/src/cuda/nvidia/NCV.hpp"
    "/home/roitberg/workspace/pcl/gpu/people/src/cuda/nvidia/NCVHaarObjectDetection.hpp"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "pcl_gpu_people")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/roitberg/workspace/pcl/gpu/people/tools/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

