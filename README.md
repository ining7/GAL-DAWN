# SC2023

0. Before getting started

Depending on your GPU, you may also want to edit the CMAKE_CUDA_ARCHITECTURES variable in $PROJECT_ROOT/CMakeLists.txt

```c++
export PROJECT_ROOT="to_your_project_path"
```

1. Modify $PROJECT_ROOT/CMakeLists.txt

According to your GPU, we use RTX 3080ti for computing, so CMAKE_CUDA_ARCHITECTURES is set to 86

```c++
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_xx,code=sm_xx")
```

Set the CMAKE_CUDA_COMPILER to the path of your NVCC, for example, "/usr/local/cuda-12.0/bin/nvcc"

```c++
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.0/bin/nvcc")
```

2.Download testing data

Unzip the compressed package and put it in the directory you need

The input data can be found on the Science Data Bank

```c++
URL=https://www.scidb.cn/s/6BjM3a
GRAPH_DIR="to_your_graph_path"
```

2. RUN

```c++
cd $PROJECT_ROOT
mkdir build
cd build
cmake .. && make -j
```

If compilation succeeds without errors, you can run your code as before, for example

```c++
cd $PROJECT_ROOT/build
./dawn_gpu $GRAPH_DIR/mouse_gene.mtx ../outpu.txt
./dawn_cpu_v2 $GRAPH_DIR/mouse_gene.mtx ../outpu.txt
```
3.Using script. 

```c++
cd ..
sudo vim ./process.sh
MAIN = ${main}
GRAPH_DIR = ${test_graph}
OUTPUT= ${outputfile}
LOG_DIR= ${GRAPH_DIR}/log
ESC && wq
sudo chmod +x ../process.sh 
sudo bash ../process.sh
```
Please note that the normal operation of the batch script needs to ensure that the test machine meets the minimum requirements. Insufficient memory or GPU memory needs to be manually adjusted according to amount of resources.

```c++
CPU: Multi-threaded processor supporting OpenMP API
RAM: 80GB or more
GPU: NVIDIA graphics cards supporting above CUDA 11.0
OS:  Ubuntu 20.04 and above
```

