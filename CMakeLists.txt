cmake_minimum_required(VERSION 2.8.3)
project(fiesta)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Use modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        nav_msgs
        visualization_msgs
        tf
        cv_bridge
        )

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

catkin_package(
  INCLUDE_DIRS include
)


include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(
    include
    SYSTEM
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

# Compiler flags per configuration
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

#add_executable(test_esdf_map
#        include/ESDFMap.h
#        src/ESDFMap.cpp
#        include/timing.h
#        src/timing.cc
#        test/test_ESDF_Map.cpp
#        )
#target_link_libraries(test_esdf_map
#        ${catkin_LIBRARIES}
#        )
#FIND_PACKAGE( OpenMP REQUIRED)
#if(OPENMP_FOUND)
#message("OPENMP FOUND")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

add_executable(test_fiesta
        include/parameters.h
        src/parameters.cpp
        include/timing.h
        src/timing.cc
        include/raycast.h
        src/raycast.cpp
        include/ESDFMap.h
        src/ESDFMap.cpp
        include/Fiesta.h
        test/test_fiesta.cpp
        )
target_link_libraries(test_fiesta
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        )

# Standalone ESDF demo with debug symbols
add_executable(esdf_standalone
        include/parameters.h
        src/parameters.cpp
        include/timing.h
        src/timing.cc
        include/raycast.h
        src/raycast.cpp
        include/ESDFMap.h
        src/ESDFMap.cpp
        src/esdf_standalone.cpp
        )

target_link_libraries(esdf_standalone
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        )

