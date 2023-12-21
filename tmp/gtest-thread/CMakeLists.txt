cmake_minimum_required(VERSION 3.15)
project(thread_demo)

file(GLOB SRC ./*.cpp)
add_executable(thread_demo_test ${SRC})
target_link_libraries(thread_demo_test pthread gtest)
