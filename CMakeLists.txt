cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言
project(dawn LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置文件根目录
set(CMAKE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(algorithm)
