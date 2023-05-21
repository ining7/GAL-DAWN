cmake_minimum_required(VERSION 3.10)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 查找MPI
find_package(MPI REQUIRED)
set(MPI_INCLUDE_PATH "/usr/include/openmpi-x86_64")
set(MPI_LIBRARIES "/usr/lib/x86_64-linux-gnu/openmpi/lib")

# 设置项目名称和语言
project(dawn_gpu_v1 LANGUAGES CXX CUDA)
project(dawn_cpu_v1 LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件路径
include_directories(include)

# 添加可执行目标
add_executable(dawn_cpu_v1 apsp/dawn_cpu_v1.cu include/dawn.cuh include/access.h)
add_executable(dawn_gpu_v1 apsp/dawn_gpu_v1.cu include/dawn.cuh include/access.h)

# 设置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# 添加MPI编译器和选项
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")

# 添加OpenMP支持
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    target_link_libraries(dawn_gpu_v1 PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dawn_cpu_v1 PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(dawn_gpu_v1 ${MPI_LIBRARIES})
target_link_libraries(dawn_cpu_v1 ${MPI_LIBRARIES})
