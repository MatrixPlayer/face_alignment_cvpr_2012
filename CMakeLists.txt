#------------------------------------------------------------------------------
# Tell cmake where to find the project cmake scripts
#------------------------------------------------------------------------------
PROJECT(cvpr_2012)
cmake_minimum_required(VERSION 2.8)

#-- Where to look first for cmake modules
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

#-- Type of build
SET(CMAKE_BUILD_TYPE Release)

FIND_PACKAGE(OpenCV REQUIRED)
MESSAGE(STATUS "OPENCV_LIBRARIES=${OpenCV_LIBS}")
MESSAGE(STATUS "OPENCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")

FIND_PACKAGE(Boost REQUIRED COMPONENTS serialization system filesystem thread REQUIRED)
MESSAGE(STATUS "BOOST_LIBRARIES=${Boost_LIBRARIES}")
MESSAGE(STATUS "BOOST_INCLUDE_DIRS=${Boost_INCLUDE_DIR}")

#-- Find include files
INCLUDE_DIRECTORIES(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

#-- Executable
ADD_EXECUTABLE(test_cvpr_2012
    ${CMAKE_SOURCE_DIR}/include/trace.hpp
    ${CMAKE_SOURCE_DIR}/include/ThreadPool.hpp
    ${CMAKE_SOURCE_DIR}/include/Timing.hpp
    ${CMAKE_SOURCE_DIR}/include/Constants.hpp
    ${CMAKE_SOURCE_DIR}/include/Viewer.hpp
    ${CMAKE_SOURCE_DIR}/src/Viewer.cpp
    ${CMAKE_SOURCE_DIR}/include/TreeNode.hpp
    ${CMAKE_SOURCE_DIR}/include/SplitGen.hpp
    ${CMAKE_SOURCE_DIR}/include/Tree.hpp
    ${CMAKE_SOURCE_DIR}/include/Forest.hpp
    ${CMAKE_SOURCE_DIR}/include/FeatureChannelFactory.hpp
    ${CMAKE_SOURCE_DIR}/include/ThresholdSplit.hpp
    ${CMAKE_SOURCE_DIR}/include/opencv_serialization.hpp
    ${CMAKE_SOURCE_DIR}/include/ImageSample.hpp
    ${CMAKE_SOURCE_DIR}/src/ImageSample.cpp
    ${CMAKE_SOURCE_DIR}/include/HeadPoseSample.hpp
    ${CMAKE_SOURCE_DIR}/src/HeadPoseSample.cpp
    ${CMAKE_SOURCE_DIR}/include/MPSample.hpp
    ${CMAKE_SOURCE_DIR}/src/MPSample.cpp
    ${CMAKE_SOURCE_DIR}/include/MeanShift.hpp
    ${CMAKE_SOURCE_DIR}/include/FaceForest.hpp
    ${CMAKE_SOURCE_DIR}/src/FaceForest.cpp
    ${CMAKE_SOURCE_DIR}/include/face_utils.hpp
    ${CMAKE_SOURCE_DIR}/src/face_utils.cpp
    
    ${CMAKE_SOURCE_DIR}/src/test_cvpr_2012.cpp
    #${CMAKE_SOURCE_DIR}/src/demo.cpp
    #${CMAKE_SOURCE_DIR}/src/train_headpose.cpp
    #${CMAKE_SOURCE_DIR}/src/eval_headpose.cpp
    #${CMAKE_SOURCE_DIR}/src/train_ffd.cpp
    #${CMAKE_SOURCE_DIR}/src/eval_ffd.cpp
)

#-- Link the executable to the libraries
TARGET_LINK_LIBRARIES(test_cvpr_2012
    ${OpenCV_LIBS}   
    jpeg
    tiff
    png
    ${Boost_LIBRARIES}
)
