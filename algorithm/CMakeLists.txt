# Add subdirectories conditionally
add_subdirectory(cpu)
find_program(NVCC_EXECUTABLE nvcc)
if(NVCC_EXECUTABLE)
    add_subdirectory(gpu)
else()
    message(STATUS "NVIDIA CUDA Compiler (NVCC) not found. Building CPU version only.")
endif()
