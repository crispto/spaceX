cmake_minimum_required(VERSION 3.5)
project(cuda_demo LANGUAGES CXX CUDA)
# find all .cpp file
set(CMAKE_CXX_FLAGS "-Wno-dev")
include_directories("$ENV{CUDA_HELP_INC}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
# find cuda
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread log)

# settings for compie ###################################

# 为了 cuda-gdb 能debug核函数
set(CMAKE_CUDA_FLAGS "-G")
# set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# stop default compile optimize
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")

# INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
message(NOTICE "Boost_LIBRARIES is ${Boost_LIBRARIES}")
message(NOTICE "Boost_INCLUDE_DIR is ${Boost_INCLUDE_DIR}")
message(NOTICE "CUDA_LIBRARIES IS ${CUDA_LIBRARIES}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ############## build all targets ########################
set(DEMO geem)
aux_source_directory(${DEMO} SOURCE)
add_executable(${DEMO} ${SOURCE})
target_link_libraries(${DEMO} ${CUDA_LIBRARIES} Boost::log co)

set(DEMO reduce)
unset(SOURCE)
aux_source_directory(${DEMO} SOURCE)
add_executable(${DEMO} ${SOURCE})
target_link_libraries(${DEMO} ${CUDA_LIBRARIES} Boost::log)

add_executable(basic basic.cpp)
target_link_libraries(basic ${CUDA_LIBRARIES})
