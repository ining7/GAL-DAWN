# Specify source and header files
file(GLOB SSSP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/algorithm/gpu/*.cu" "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB SSSP_HEADERS "${CMAKE_SOURCE_DIR}/include/algorithm/gpu/*.cuh" "${CMAKE_SOURCE_DIR}/include/*.hxx" "${CMAKE_SOURCE_DIR}/include/dawn.cuh")

# Define the CUDA sources
set(SSSP_GPU_SOURCES
    sssp_gpu.cu
    ${SSSP_SOURCES}
)

# Add the executable
add_executable(sssp_gpu ${SSSP_GPU_SOURCES})
target_include_directories(sssp_gpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Set properties for CUDA
set_property(TARGET sssp_gpu PROPERTY CUDA_STANDARD 14)
set_property(TARGET sssp_gpu PROPERTY CUDA_ARCHITECTURES 86)

# Add compile options for CUDA
target_compile_options(sssp_gpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -fopenmp -O3
>)
set_target_properties(sssp_gpu PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sssp_gpu PUBLIC OpenMP::OpenMP_CXX)
endif()

