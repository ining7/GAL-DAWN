# Specify source and header files
file(GLOB MSSP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/algorithm/gpu/*.cu" "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB MSSP_HEADERS "${CMAKE_SOURCE_DIR}/include/algorithm/gpu/*.cuh" "${CMAKE_SOURCE_DIR}/include/*.hxx" "${CMAKE_SOURCE_DIR}/include/dawn.cuh")

# Define the CUDA sources
set(MSSP_GPU_SOURCES
    mssp_gpu.cu
    ${MSSP_SOURCES}
)

# Add the executable
add_executable(mssp_gpu ${MSSP_GPU_SOURCES})
target_include_directories(mssp_gpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Set properties for CUDA
set_property(TARGET mssp_gpu PROPERTY CUDA_STANDARD 14)
set_property(TARGET mssp_gpu PROPERTY CUDA_ARCHITECTURES 86)

# Add compile options for CUDA
target_compile_options(mssp_gpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -fopenmp -O3
>)
set_target_properties(mssp_gpu PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mssp_gpu PUBLIC OpenMP::OpenMP_CXX)
endif()

