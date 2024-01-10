# Specify source and header files
file(GLOB bfs_SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB SSSP_HEADERS "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.hxx")

# Define the CUDA sources
set(bfs_CPU_SOURCES
    bfs_cpu.cpp
    ${bfs_SOURCES_CPP}
)

# Add the executable
add_executable(bfs_cpu ${bfs_CPU_SOURCES})
target_include_directories(bfs_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Add compile options
target_compile_options(bfs_cpu PUBLIC -O3 -fopenmp)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bfs_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

