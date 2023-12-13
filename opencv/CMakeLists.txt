# find openv
set(CMAKE_BUILD_TYPE Debug)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "OpenCV found")
  message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
  message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
else()
  message(FATAL_ERROR "OpenCV not found")
endif()

add_executable(perf_demo perf_opencv_demo.cpp)
target_link_libraries(perf_demo ${OpenCV_LIBS})


option(BUILD_GPERF "Build gperf demo" OFF)
if(BUILD_GPERF)
  add_executable(gperf_demo gperf_opencv_demo.cpp)
  target_link_libraries(gperf_demo${OpenCV_LIBS} profiler )
endif()
