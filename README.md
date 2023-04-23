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
./dawn_cpu_v3 $GRAPH_DIR/mouse_gene.mtx ../outpu.txt
./dawn_cpu_v4 $GRAPH_DIR/mouse_gene.mtx ../outpu.txt
./dawn_cpu_big $GRAPH_DIR/graph_CRC.txt $GRAPH_DIR/graph_RCC.txt ../outpu.txt
./convert $GRAPH_DIR/large_graph.mtx $GRAPH_DIR/graph_CRC.txt $GRAPH_DIR/graph_RCC.txt
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
RAM: 8GB or more
GPU: NVIDIA graphics cards supporting above CUDA 11.0
OS:  Ubuntu 20.04 and above
```
4.Release version

For the CPU version, dawn_cpu_v3 is fine-grained parallel version and dawn_cpu_v4 is the coarse-grained parallel version. The fine-grained parallel version of DAWN only requires the path statistics at the end of each loop to be executed in serial, while the coarse-grained parallel version has no serial phase and the data between threads are completely independent.

For the large-scale graph, you can use dawn_cpu_big, which is the version for large-scale graph, and you need use the convert tool to process the graph first. Convert tool will compress the large-scale graph to graph_CRC.txt and graph_RCC.txt, which is the inputfile of the dawn_cpu_big.

4.Release result

On the test machine with i5-13600KF, dawn_cpu_v3 and dawn_cpu_v4 achieves average speedup of 1.857x and 6.423x over GDS, respectively. On the 64-thread AMD EPYC 7T83, various version of DAWN achieved speedups of 1.738x and 1.579x, over running on the 20-thread i5-13600KF.

On the test machine with i5-13600KF, dawn_cpu_big requires about 155 hours to process Graph kmer_V1r, which has 214,005,017 nodes and 465,410,904 edges.

5.New version
The version of DAWN with better performance in sparse graph is under development, which has lower time complexity.
We will release GPU_v2 as soon as possible.
