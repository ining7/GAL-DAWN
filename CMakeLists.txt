cmake_minimum_required(VERSION 3.10)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.0/bin/nvcc")

# 设置项目名称和语言
project(dawn LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(dawn_gpu_v1 LANGUAGES CXX CUDA)
project(dawn_cpu_v1 LANGUAGES CXX CUDA)
project(dawn_cpu_sssp LANGUAGES CXX CUDA)
project(dawn_gpu_sssp LANGUAGES CXX CUDA)
project(convert LANGUAGES CXX CUDA)

# 添加头文件路径
include_directories(include)

# 添加可执行目标
add_executable(dawn_cpu_v1 apsp/dawn_cpu_v1.cu include/dawn.cuh include/access.h)
add_executable(dawn_gpu_v1 apsp/dawn_gpu_v1.cu include/dawn.cuh include/access.h)
add_executable(dawn_cpu_sssp sssp/dawn_cpu_sssp.cu include/dawn.cuh include/access.h)
add_executable(dawn_gpu_sssp sssp/dawn_gpu_sssp.cu include/dawn.cuh include/access.h)
add_executable(convert tool/convert.cu include/dawn.cuh include/access.h)

# 设置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# 添加OpenMP支持
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    target_link_libraries(dawn_cpu_sssp PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dawn_gpu_sssp PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dawn_gpu_v1 PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dawn_cpu_v1 PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(convert PUBLIC OpenMP::OpenMP_CXX)
endif()


