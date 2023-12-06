cmake_minimum_required(VERSION 3.16)
project(cuda_sample LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type -Wreturn-type")

include_directories(${CUDA_HELP_INC})
# find cuda
find_package(CUDA REQUIRED)
message(NOTICE "$CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(NOTICE "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
file(GLOB demo_dir *.cpp)

include_directories(${CUDA_INCLUDE_DIRS})

# set_target_properties( ${subdir_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
# ${PROJECT_BINARY_DIR}/bin/cuda)

add_executable(reduce reduction.cpp reduction.cu)
target_link_libraries(reduce ${CUDA_LIBRARIES} ${Boost_LIBRARIES})
