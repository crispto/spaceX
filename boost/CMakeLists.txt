cmake_minimum_required(VERSION 3.5)
project(boost_demo)

set(CMAKE_CXX_STANDARD 11)

# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS system thread log)
# print all boost variables
message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
message(STATUS "Boost_LIBRARY_DIRS = ${Boost_LIBRARY_DIRS}")
message(STATUS "Boost_VERSION = ${Boost_VERSION}")

# Add bind_demo executable
add_executable(bind_demo bind_demo.cpp)
# Link with all Boost libraries
target_link_libraries(bind_demo ${Boost_LIBRARIES})

add_executable(circle_demo circle_demo.cpp)
target_link_libraries(circle_demo ${Boost_LIBRARIES})
