cmake_minimum_required(VERSION 3.10)
project(tensorrt_demo)
set(CMAKE_CXX_STANDARD 11)
# ~~~
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "OpenCV found")
  message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "OpenCV_LIBS:
${OpenCV_LIBS}")
  message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIB_DIR})
else()
  message(FATAL_ERROR "OpenCV not found")
endif()
# ~~~

# ------------------- common -------------------
# 公共库和引入的头文件
# ------------------- common -------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

set(CUDART_LIB cudart)
set(CUDNN_LIB cudnn)
set(CUBLAS_LIB cublas)
set(NVINFER_LIB nvinfer)
set(NVPARSERS_LIB nvparsers)
set(NVINFER_PLUGIN_LIB nvinfer_plugin)
set(NVONNXPARSERS_LIB nvonnxparser)

include_directories("/usr/local/include" "/usr/local/cuda/include"
                    "/usr/local/tensorrt/include")
link_directories("/usr/local/lib" "/usr/local/cuda/lib64"
                 "/usr/local/tensorrt/lib" common_lib)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()
set(NV_LIBS
    ${NVINFER_LIB}
    ${NVPARSERS_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${NVONNXPARSERS_LIB}
    ${CUDART_LIB}
    ${CUDNN_LIB}
    ${CUBLAS_LIB})
    set(NV_LIBS ${NV_LIBS} ${CUDA_LIBRARIES})
message(NOTICE "CUDA_VERSION: ${CUDA_VERSION}")

set(TRT_HELP tensorrt_help_${CUDA_VERSION})
set(EXTRA_LIBS rt dl pthread co ${TRT_HELP})

add_definitions(-DCUDA_VERSION=${CUDA_VERSION})
# ~~~
# lib
# file(GLOB COMMON_SRC common/*.cpp)

# add_library(${TRT_HELP} SHARED ${COMMON_SRC})

# target_link_libraries(${TRT_HELP} ${NV_LIBS})
# ~~~
# ------------------- demo -------------------
aux_source_directory(samples/ SRC)
foreach(demo ${SRC})
  get_filename_component(demo_name ${demo} NAME_WE)
  add_executable(${demo_name} ${demo})
  target_link_libraries(${demo_name} co ${NV_LIBS} ${EXTRA_LIBS} ${OpenCV_LIBS})
endforeach()

# set_target_properties( ${demo_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
# ${PROJECT_BINARY_DIR}/bin/tensorrt)
