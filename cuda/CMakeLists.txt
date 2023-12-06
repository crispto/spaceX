cmake_minimum_required(VERSION 3.5)
project(cuda_demo LANGUAGES CXX CUDA)
# find all .cpp file
set(CUDA_HELP_INC
    "$ENV{CUDA_SAMPLE_DATA}/Common")
message(NOTICE "CUDA_HELP_INC: ${CUDA_HELP_INC}")
include_directories(${CUDA_HELP_INC})
# find cuda
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA REQUIRED)

add_executable(basic basic.cpp)
target_link_libraries(basic ${CUDA_LIBRARIES})


set(CMAKE_CUDA_FLAGS "-G")
# set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
aux_source_directory(geem GEEM_SRC)
add_executable(geem ${GEEM_SRC})
target_link_libraries(geem ${CUDA_LIBRARIES})

