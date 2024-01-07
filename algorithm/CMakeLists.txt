# Find CUDA package
find_package(CUDAToolkit)

# Add subdirectories conditionally
add_subdirectory(apsp_cpu)
add_subdirectory(mssp_cpu)
add_subdirectory(sssp_cpu)
if(CUDAToolkit_FOUND)
    add_subdirectory(apsp_gpu)
    add_subdirectory(mssp_gpu)
    add_subdirectory(sssp_gpu)
endif()
