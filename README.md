# SC2023

0. Before getting started

Depending on your GPU, you may also want to edit the CMAKE_CUDA_ARCHITECTURES variable in $PROJECT_ROOT/CMakeLists.txt

export PROJECT_ROOT="to_your_project_path"

1. Modify $PROJECT_ROOT/CMakeLists.txt

According to your GPU, we use RTX 3080ti for computing, so CMAKE_CUDA_ARCHITECTURES is set to 86

set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_xx,code=sm_xx")

Set the CMAKE_CUDA_COMPILER to the path of your NVCC, for example, "/usr/local/cuda-12.0/bin/nvcc"

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.0/bin/nvcc")

2.Download testing data

Unzip the compressed package and put it in the directory you need

The input data can be found on the Science Data Bank

URL=https://www.scidb.cn/s/6BjM3a
GRAPH_DIR="to_your_graph_path"

2. RUN

cd $PROJECT_ROOT
mkdir build
cd build
cmake .. && make -j

If compilation succeeds without errors, you can run your code as before, for example

cd $PROJECT_ROOT/build
./dawn_gpu $GRAPH_DIR/mouse_gene.mtx ../outpu.txt
./dawn_cpu_v2 $GRAPH_DIR/mouse_gene.mtx ../outpu.txt
