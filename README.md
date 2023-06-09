# DAWN V2.0

DAWN is a novel shortest paths algorithm, which is suitable for weighted and unweighted graphs. DAWN requires $O(m)$ space and $O(S_{wcc} \cdot E_{wcc})$ times on the unweighted graphs, which can also process SSSP tasks and requires $O(m)$ and $O(E_{wcc})$ time on the connected and unconnected graphs. $S_{wcc}$ and $E_{wcc}$ denote the number of nodes and edges included in the largest WCC (Weakly Connected Component) in the graphs.

At parent, DAWN cannot run on the graph with negative weighted. If we determine after a thorough investigation that DAWN cannot handle negative-weighted graphs, we will inform you accordingly.  

## Quick Start Guide

### 0. Before getting started

Depending on your GPU, you may also want to edit the CMAKE_CUDA_ARCHITECTURES variable in $PROJECT_ROOT/CMakeLists.txt

```c++
export PROJECT_ROOT="to_your_project_path"
```

### 1. Modify $PROJECT_ROOT/CMakeLists.txt

According to your GPU, we use RTX 3080ti for computing, so CMAKE_CUDA_ARCHITECTURES is set to 86

```c++
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_xx,code=sm_xx")
```

Set the CMAKE_CUDA_COMPILER to the path of your NVCC, for example,

```c++
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
```

### 2.Download testing data

Unzip the compressed package and put it in the directory you need

The input data can be found on the Science Data Bank

```c++
URL=https://www.scidb.cn/s/6BjM3a
GRAPH_DIR="to_your_graph_path"
```

### 3. Build and RUN

```c++
cd $PROJECT_ROOT
mkdir build && cd build
cmake .. && make -j
```

If compilation succeeds without errors, you can run your code as before, for example

```c++
cd $PROJECT_ROOT/build
./dawn_cpu_v1 SG $GRAPH_DIR/mouse_gene.mtx ../output.txt 100 false 0 unweighted

./dawn_cpu_v1 SG $GRAPH_DIR/cage10.mtx ../output.txt 100 false 0 weighted

./dawn_gpu_v1 Default $GRAPH_DIR/mouse_gene.mtx ../output.txt 4 256 100 false 0

```

When the version is built, it will also prepare SSSP applications, which can be used directly.

If you need to use DAWN in your own solution, please check the source code under the **sssp** folder and call it.

If you do not have the conditions to use NVCC, you need to comment out all GPU-related statements in **Cmakelists.txt**, then use GCC or clang to build applications that can only run on the cpu. (GCC 9.4.0 and above, clang 10.0.0 and above)

### 4.Using script

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

### For general graphs

```c++
CPU: Multi-threaded processor supporting OpenMP API
RAM: 4GB or more
GPU: 1GB or more
Compiler: NVCC of CUDA 11.0 above
OS:  Ubuntu 20.04 and above
```

### For large-scale graphs

```c++
CPU: Multi-threaded processor supporting OpenMP API
RAM: 10GB or more
GPU: 4GB or more
Compiler: NVCC of CUDA 11.0 above
OS:  Ubuntu 20.04 and above
```

### 5.Release version

| Version | Implementation |
| ------ | ------ |
| APSP/TG |  Version of Thread Parallel|
| APSP/SG |  Version of Stream Parallel|
|MSSP/S| Version of Multi-source via Thread Parallel|
|MSSP/P| Version of Multi-source via Stream Parallel|
|SSSP| Version of Single-source|

Currently, on the Single-Processor, SG allocates one thread per stream. On Multi-Processor, SG will allocate multiple threads per stream. (It will be supported in the next version.)

For the GPU version, you can use Default, please make sure that there is enough GPU memory for the graph.

```c++
int device;
cudaDeviceProp props;
cudaGetDevice(&device);
cudaGetDeviceProperties(&props, device);
printf("Max shared memory per block: %ld\n", props.sharedMemPerBlock);
```

If you are sure that the size of the thread block and the scale of the graph is set reasonably according to the device parameters.

### 6.Release result

On the test machine with i5-13600KF, SG achieves average speedup of 1212.524&times; and 1,315.953&times; over BFS and SSSP from Gunrock (hereinafter referred to as GDS and BFS), respectively. On the test machine with i5-13600KF, DAWN need more than 10GB free memory to solve the large graph [281K,214M].

On the 64-thread AMD EPYC 7T83, DAWN achieved speedups of 3.532945 &times;, 3862.372&times; and 3557.543&times;, over DAWN(20), GDS and BFS, respectively.

On the test machine with RTX3080TI, DAWN achieves average speedup of 2.779&times; 4.534&times; and 4.173&times;, over SG, GDS and BFS, respectively.

We provide the file **check_unweighted.py** and **check_weighted.py**, based on networkx, which can be used to check the results printed by DAWN.

### 7.Decumentation

Please refer to [decument/Decumentation_v1](https://github.com/lxrzlyr/SC2023/blob/eb9080f76c2950981a4dac72141d4991eff8b9db/document/Decumentation_v1.md) for commands.

## New version

The version of DWAN on the weighted graph has been included in DAWN V2.0. The Artifacts runs on the weighted graphs via SOVM implementation, hence, we cannot commit to a specific timeline for the release of a GPU version of SOVM running on weighted graphs. Currently, we believe that the implementation of SOVM is not suitable for running on GPUs without a thorough investigation of parallel methods on GPUs. However, the GPU version of the DAWN based on the SOVM implementation has always been in our plan.

In the future, we plan to develop more algorithms based on DAWN, including but not limited to Between Centrality, Closeness Centrality, etc. Further applications of these algorithms, such as community detection, clustering, and path planning, are also on our agenda.

We welcome any interest and ideas related to DAWN and its applications. If you are interested in DAWN algorithms and their applications, please feel free to share your thoughts via email [<1289539524@qq.com>], and we will do our best to assist you in your research based on DAWN.

We will release new features of DAWN and the application algorithms based on DAWN on this repository. If the algorithms are also needed by Gunrock, we will contribute them to the Gunrock repository later. The DAWN component based on Gunrock may be released to the main/develop branch in the near future, so please stay tuned to the [Gunrock](https://github.com/gunrock/gunrock).

## How to Cite DAWN

Thank you for citing our work.

```bibtex
@misc{feng2023novel,
      title={A Novel Shortest Paths Algorithm on Unweighted Graphs}, 
      author={Yelai Feng and Huaixi Wang and Yining Zhu and Chao Chang and Hongyi Lu},
      year={2023},
      eprint={2208.04514},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
