# begin /* Set the application name. */
set(APPLICATION_NAME mssp)
# end /* Set the application name. */

# begin /* Add CUDA executables */
add_executable(${APPLICATION_NAME})

set(SOURCE_LIST 
    ${APPLICATION_NAME}.cu
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(OpenMP REQUIRED)
target_link_libraries(${APPLICATION_NAME} PRIVATE OpenMP::OpenMP_CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")


target_sources(${APPLICATION_NAME} PRIVATE ${SOURCE_LIST})
target_link_libraries(${APPLICATION_NAME} PRIVATE essentials)
get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
set_target_properties(${APPLICATION_NAME} 
    PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
) # XXX: Find a better way to inherit essentials properties.

message(STATUS "Example Added: ${APPLICATION_NAME}")
# end /* Add CUDA executables */