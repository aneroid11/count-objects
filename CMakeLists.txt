cmake_minimum_required(VERSION 3.24)
project(count_objects)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(count_objects main.cpp)
target_link_libraries(count_objects ${OpenCV_LIBS})
