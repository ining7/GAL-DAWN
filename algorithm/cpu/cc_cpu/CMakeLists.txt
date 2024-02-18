# Specify source and header files
file(GLOB cc_SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.hxx")

# Define the CUDA sources
set(cc_CPU_SOURCES
    cc_cpu.cpp
    ${cc_SOURCES_CPP}
)

# Add the executable
add_executable(cc_cpu ${cc_CPU_SOURCES})
target_include_directories(cc_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Add compile options
target_compile_options(cc_cpu PUBLIC -O3 -fopenmp)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cc_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

