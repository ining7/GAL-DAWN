# Specify source and header files
file(GLOB APSP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/algorithm/gpu/*.cu" "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB APSP_HEADERS "${CMAKE_SOURCE_DIR}/include/algorithm/gpu/*.cuh" "${CMAKE_SOURCE_DIR}/include/*.hxx" "${CMAKE_SOURCE_DIR}/include/dawn.cuh")

# Define the CUDA sources
set(APSP_GPU_SOURCES
    apsp_gpu.cu
    ${APSP_SOURCES}
)

# Add the executable
add_executable(apsp_gpu ${APSP_GPU_SOURCES})
target_include_directories(apsp_gpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# Set properties for CUDA
set_property(TARGET apsp_gpu PROPERTY CUDA_STANDARD 14)
set_property(TARGET apsp_gpu PROPERTY CUDA_ARCHITECTURES 86)

# Add compile options for CUDA
target_compile_options(apsp_gpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -fopenmp -O3
>)
set_target_properties(apsp_gpu PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(apsp_gpu PUBLIC OpenMP::OpenMP_CXX)
endif()

