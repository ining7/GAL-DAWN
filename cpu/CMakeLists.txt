cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言
project(dawn LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件路径
include_directories(include)

project(dawn_cpu_v1 LANGUAGES CXX)
project(dawn_cpu_sssp LANGUAGES CXX)
project(convert LANGUAGES CXX)

# 添加可执行目标
add_executable(dawn_cpu_v1 dawn_cpu_v1.cpp dawn.hpp access.h)
add_executable(dawn_cpu_sssp dawn_cpu_sssp.cpp dawn.hpp access.h)
add_executable(convert convert.cpp dawn.hpp access.h)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 添加OpenMP支持
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    target_link_libraries(dawn_cpu_sssp PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dawn_cpu_v1 PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(convert PUBLIC OpenMP::OpenMP_CXX)
endif()


