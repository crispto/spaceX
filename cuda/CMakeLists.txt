cmake_minimum_required(VERSION 3.5)
project(cuda_demo)
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
file(GLOB demo_dir *.cpp)

foreach(src ${demo_dir})
  get_filename_component(subdir_name ${src} NAME_WE)
  add_executable(${subdir_name} ${src})
  target_link_libraries(${subdir_name} ${CUDA_LIBRARIES})
  # set target bin directory bin/cuda

  # set_target_properties( ${subdir_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  # ${PROJECT_BINARY_DIR}/bin/cuda)
endforeach()
