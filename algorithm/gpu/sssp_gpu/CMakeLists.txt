# Specify source and header files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/algorithm/gpu/*.cu" "${CMAKE_SOURCE_DIR}/src/*.cu")

# Add the executable
add_executable(sssp_gpu sssp_gpu.cu ${SOURCES})
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


