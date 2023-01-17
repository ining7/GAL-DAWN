cmake_minimum_required(VERSION 3.10)
project(dense)
project(bitset)

find_package(OpenMP REQUIRED)

add_executable(bitset v_bitset.cpp timer.hpp access.h)

target_link_libraries(bitset OpenMP::OpenMP_CXX)