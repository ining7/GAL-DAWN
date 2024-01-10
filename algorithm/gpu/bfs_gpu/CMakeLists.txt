# Specify source and header files
file(GLOB SSSP_SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB SSSP_HEADERS "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.cuh" "${CMAKE_SOURCE_DIR}/include/*.hxx")

# Define the CUDA sources
set(bfs_GPU_SOURCES
    bfs_gpu.cu
    ${SSSP_SOURCES_CPP}
)

# Add the executable
add_executable(bfs_gpu ${bfs_GPU_SOURCES})
target_include_directories(bfs_gpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# Set properties for CUDA
set_property(TARGET bfs_gpu PROPERTY CUDA_STANDARD 14)
set_property(TARGET bfs_gpu PROPERTY CUDA_ARCHITECTURES 86)

# Add compile options for CUDA
target_compile_options(bfs_gpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-code arch=compute_86,code=sm_86
    -Xcompiler -fopenmp
>)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bfs_gpu PUBLIC OpenMP::OpenMP_CXX)
endif()

