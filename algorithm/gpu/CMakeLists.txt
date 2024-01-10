
project(dawn LANGUAGES CXX CUDA)

set(CUDA_ARCHITECTURES "86")

add_subdirectory(apsp_gpu)
add_subdirectory(bfs_gpu)
add_subdirectory(mssp_gpu)
add_subdirectory(sssp_gpu)