# find all .cpp file
set(CUDA_HELP_INC
    "/home/SENSETIME/yangchengwei/coding/github/cuda-samples/common/inc")
include_directories(${CUDA_HELP_INC})
# find cuda
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
