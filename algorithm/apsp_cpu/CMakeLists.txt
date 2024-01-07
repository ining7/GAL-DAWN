# Specify source and header files
file(GLOB APSP_SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB APSP_HEADERS "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.hxx")

# Define the CUDA sources
set(APSP_CPU_SOURCES
    apsp_cpu.cpp
    ${APSP_SOURCES_CPP}
)

# Add the executable
add_executable(apsp_cpu ${APSP_CPU_SOURCES})
target_include_directories(apsp_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Add compile options
target_compile_options(apsp_cpu PUBLIC -O3 -fopenmp)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(apsp_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

