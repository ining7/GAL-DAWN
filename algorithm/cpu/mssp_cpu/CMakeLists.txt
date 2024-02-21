# Specify source and header files
file(GLOB MSSP_SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/algorithm/cpu/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/algorithm/cpu/*.hxx" "${CMAKE_SOURCE_DIR}/include/*.hxx")

# Define the CUDA sources
set(mssp_CPU_SOURCES
    mssp_cpu.cpp
    ${MSSP_SOURCES_CPP}
)

# Add the executable
add_executable(mssp_cpu ${mssp_CPU_SOURCES})
target_include_directories(mssp_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Add compile options
target_compile_options(mssp_cpu PUBLIC -O3 -fopenmp)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mssp_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

