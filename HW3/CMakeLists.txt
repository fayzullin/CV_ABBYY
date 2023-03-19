cmake_minimum_required(VERSION 3.21)
project(HW3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( FHT main.cc fht.cc)
target_link_libraries( FHT ${OpenCV_LIBS} )
