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
mkdir build && cd build
cmake .. && make -j
```

If compilation succeeds without errors, you can run your code as before, for example

```c++
cd $PROJECT_ROOT/build
./dawn_cpu_v1 CG $GRAPH_DIR/mouse_gene.mtx ../outpu.txt 100 false 0

./dawn_gpu_v1 $GRAPH_DIR/mouse_gene.mtx ../outpu.txt 8 4 100 false

./dawn_cpu_v1 BCG $GRAPH_DIR/graph.mxt $GRAPH_DIR/graph_CRC.txt $GRAPH_DIR/graph_RCC.txt ../outpu.txt 10000 false 0

./convert $GRAPH_DIR/large_graph.mtx $GRAPH_DIR/graph_CRC.txt $GRAPH_DIR/graph_RCC.txt
```

When the version is built, it will SSSP applications, which can be used directly. 

Please refer to decument/Decumention_v1 for commands.

If you need to use DAWN in your own solution, please check the source code under the **sssp** folder and call it.

If you do not have the conditions to use NVCC, you can enter the **cpu** folder, use GCC or clang to build applications that can only run on the cpu. (GCC 9.4.0 and above, clang 10.0.0 and above)

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
GPU: 1GB or more
Compiler: NVCC of CUDA 11.0 above
OS:  Ubuntu 20.04 and above
```

4.Release version

For the CPU version, FG is fine-grained parallel version and CG is the coarse-grained parallel version. The fine-grained parallel version of DAWN only requires the path statistics at the end of each loop to be executed in serial, while the coarse-grained parallel version has no serial phase and the data between threads are completely independent.

For the large-scale graph, you can use BFG and BCG, which is the version for large-scale graph, and you need use the convert tool to process the graph first. Convert tool will compress the large-scale graph to graph_CRC.txt and graph_RCC.txt, which is the inputfile.

For the GPU version, you can use Default and Big, please make sure that there is enough GPU memory for the graph.

4.Release result

On the test machine with i5-13600KF, FG and CG achieves average speedup of 1.857x and 6.423x over GDS, respectively. On the 64-thread AMD EPYC 7T83, various version of DAWN achieved speedups of 1.738x and 1.579x, over running on the 20-thread i5-13600KF.

On the test machine with i5-13600KF, BFG need 10GB free memory to solve the Graph kmer_V1r with 214M nodes and 465M edges, which require average 92 minutes for 21.4K nodes in the graph and average 0.257 seconds for SSSP. For Graph wiki-Talk with 2.39M nodes and 5M egdes, DAWN can compelet work of APSP problem in 1475 secconds. We hope that our work can make it possible for everyone to widely use personal computers to analyze the graphs over 200M nodes, although at present we need a little patience to wait for the results.

On the test machine with RTX3080TI, dawn_gpu_v1 achieves average speedup of 6.336x, 1.509X and 5.291x over FG, CG and GDS.

5.New version
The further optimization of DAWN has achieved a theoretical breakthrough, and we will start making new artifacts as soon as possible. DAWN2.0 will have better performance in sparse graph and with lower time complexity.

The version of DWAN on the weighted graph will be include in DAWN2.0.
