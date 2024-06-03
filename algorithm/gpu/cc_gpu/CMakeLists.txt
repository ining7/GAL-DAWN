# Specify source and header files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/algorithm/gpu/*.cu" "${CMAKE_SOURCE_DIR}/src/*.cu")

# Add the executable
add_executable(cc_gpu cc_gpu.cu ${SOURCES})
target_include_directories(cc_gpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Set properties for CUDA
set_property(TARGET cc_gpu PROPERTY CUDA_STANDARD 14)
set_property(TARGET cc_gpu PROPERTY CUDA_ARCHITECTURES 86)

# Add compile options for CUDA
target_compile_options(cc_gpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -fopenmp -O3
>)
set_target_properties(cc_gpu PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cc_gpu PUBLIC OpenMP::OpenMP_CXX)
endif()

