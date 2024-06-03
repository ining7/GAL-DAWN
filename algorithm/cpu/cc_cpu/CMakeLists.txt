# Specify source
file(GLOB SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/algorithm/cpu/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Add the executable
add_executable(cc_cpu cc_cpu.cpp ${SOURCES_CPP})

# Include directories
target_include_directories(cc_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Compile options
target_compile_options(cc_cpu PUBLIC -O3)

# Find and link OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_compile_options(cc_cpu PUBLIC -fopenmp)
    target_link_libraries(cc_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

