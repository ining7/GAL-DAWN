# Specify source and header files
file(GLOB sssp_SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.hxx")

# Define the CUDA sources
set(sssp_CPU_SOURCES
    sssp_cpu.cpp
    ${sssp_SOURCES_CPP}
)

# Add the executable
add_executable(sssp_cpu ${sssp_CPU_SOURCES})
target_include_directories(sssp_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Add compile options
target_compile_options(sssp_cpu PUBLIC -O3 -fopenmp)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sssp_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

