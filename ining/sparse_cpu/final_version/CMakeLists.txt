cmake_minimum_required(VERSION 3.10)
project(final_versi)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)

add_executable(Sparse sparse.cpp access.h)

target_link_libraries(Sparse OpenMP::OpenMP_CXX)