cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.0/bin/nvcc")

# 设置项目名称
project(dawn LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(dawn_cpu LANGUAGES CXX CUDA)
project(dawn_gpu LANGUAGES CXX CUDA)
project(dawn_cpu_v2 LANGUAGES CXX CUDA)
project(dawn_cpu_v3 LANGUAGES CXX CUDA)

# cmake_policy(SET CMP0104 NEW)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

add_executable(dawn_cpu dawn_cpu.cu access.h timer.hpp)
add_executable(dawn_gpu dawn_gpu.cu access.h)
add_executable(dawn_cpu_v2 dawn_cpu_v2.cu access.h)
add_executable(dawn_cpu_v3 dawn_cpu_v3.cu access.h)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    target_link_libraries(dawn_cpu PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dawn_gpu PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dawn_cpu_v2 PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dawn_cpu_v3 PUBLIC OpenMP::OpenMP_CXX)
endif()

# Find CUDA library
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    target_link_libraries(dawn_gpu PUBLIC cublas)
endif()
