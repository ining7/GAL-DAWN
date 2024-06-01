# Add subdirectories conditionally
find_program(NVCC_EXECUTABLE nvcc)
add_subdirectory(cpu)
if(NVCC_EXECUTABLE)
    message(STATUS "NVCC found: ${NVCC_EXECUTABLE}")
    # add_subdirectory(gpu)
else()
    message(STATUS "NVCC not found, only building CPU version")
endif()
