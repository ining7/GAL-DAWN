cmake_minimum_required(VERSION 3.10)

project(demo LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

find_package(OpenMP REQUIRED)

add_executable(demo demo.cu access.h timer.hpp)

set_property(TARGET demo PROPERTY CUDA_SEPARABLE_COMPILATION ON)   

target_link_libraries(demo OpenMP::OpenMP_CXX)