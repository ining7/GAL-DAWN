# Find CUDA package
find_package(CUDAToolkit)

# Add subdirectories conditionally
add_subdirectory(cpu)
if(CUDAToolkit_FOUND)
    add_subdirectory(gpu)
endif()
