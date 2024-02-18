# DAWN V2.2

DAWN is a novel shortest paths algorithm, which is suitable for weighted and unweighted graphs. DAWN requires $O(m)$ space and $O(S_{wcc} \cdot E_{wcc})$ times on the unweighted graphs, which can also process SSSP tasks and requires $O(E_{wcc}(i))$ time. $S_{wcc}$ and $E_{wcc}$ denote the number of nodes and edges included in the largest WCC (Weakly Connected Component) in the graphs.

DAWN is capable of solving the APSP and SSSP problems on graphs with negative weights, and can automatically exclude the influence of negative weight cycles.  

## Quick Start Guide

### 0. Before getting started

Depending on your GPU, you may also want to edit the CUDA_ARCHITECTURES variable in $PROJECT_ROOT/algorithm/gpu/CMakeLists.txt

```c++
export PROJECT_ROOT="to_your_project_path"
```

### 1. Modify $PROJECT_ROOT/CMakeLists.txt

According to your GPU, we use RTX 3080ti for computing, so CUDA_ARCHITECTURES is set to 86

```c++
set(CUDA_ARCHITECTURES "86")
```

If the machine does not have a GPU available, DAWN will automatically detect the CUDA environment and only build artifacts for the CPU. However, if you are certain that the machine has a usable GPU and you have been unable to build artifacts for the GPU correctly, we suspect there may be an issue with the CUDA environment. Please further check for any path-related problems.

### 2.Download testing data

Unzip the compressed package and put it in the directory you need

The input data can be found on the Science Data Bank

```c++
URL=https://www.scidb.cn/s/6BjM3a
GRAPH_DIR="to_your_graph_path"
```

### 3. Dependencies

DAWN builds, runs, and has been tested on Ubuntu/Rocky Linux. Even though DAWN may work on other linux systems, we have not tested correctness or performance. DAWN is not available on Windows and cannot achieve the same performance on WSL systems. Please beware.

At the minimum, DAWN depends on the following software:

- A modern C++ compiler compliant with the C++ 14 standard
- GCC (>= 9.4.0 or Clang >= 10.0.0)
- CMake (>= 3.10)
- libomp (>= 10.0)

If you need run DAWN on the GPU, expand:

- CUDA (>= 11.0)
- thrust (>= 2.0)

### 4. Build and RUN

```c++
cd $PROJECT_ROOT
mkdir build && cd build
cmake .. && make -j
```

If compilation succeeds without errors, you can run your code as before, for example

```c++
cd $PROJECT_ROOT/build
./apsp_cpu SG $GRAPH_DIR/mouse_gene.mtx ../output.txt false 0 unweighted

./apsp_cpu SG $GRAPH_DIR/cage10.mtx ../output.txt false 0 weighted

```

When the version is built, it will also prepare SSSP and MSSP applications, which can be used directly.

If you need to use DAWN in your own solution, please check the source code under the **src**，**include** folder and call them.

#### Using script

```c++
cd ..
vim ./process.sh
MAIN = ${main}
GRAPH_DIR = ${test_graph}
OUTPUT= ${outputfile}
LOG_DIR= ${GRAPH_DIR}/log
ESC && wq
sudo chmod +x ../process.sh
bash ../process.sh
```

Please note that the normal operation of the batch script needs to ensure that the test machine meets the minimum requirements. Insufficient memory or GPU memory needs to be manually adjusted according to amount of resources.

#### For general graphs

```c++
CPU: Multi-threaded processor supporting OpenMP API
RAM: 8GB or more
GPU: 1GB or more
Compiler: NVCC of CUDA 11.0 above
OS:  Ubuntu 20.04 and above
```

#### For large-scale graphs

```c++
CPU: Multi-threaded processor supporting OpenMP API
RAM: 24GB or more
GPU: 4GB or more
Compiler: NVCC of CUDA 11.0 above
OS:  Ubuntu 20.04 and above
```

### 5.Release version

| Algorithm | Implementation | Weigthed |
| ------ | ------ | ------ |
| APSP |  GOVM| True  |
| APSP |  SOVM| False |
| MSSP |  GOVM| True  |
| MSSP |  SOVM| False |
| SSSP |  GOVM| True  |
| BFS  |  SOVM| False |

For the GPU version, please make sure that there is enough GPU memory for the graph. The size of the thread block and the scale of the graph is set reasonably according to the device parameters.

```c++
int device;
cudaDeviceProp props;
cudaGetDevice(&device);
cudaGetDeviceProperties(&props, device);
printf("Max shared memory per block: %ld\n", props.sharedMemPerBlock);
```

### 6.Performance

We have presented a performance comparison of algorithms for DAWN, GAPBS, and Gunrock in a [table](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/blob/dev/test/performance.md). The benchmark tests were run on the Gunrock benchmark dataset and the Suite Sparse Collection dataset. The table provides specific information about the graphs and their corresponding runtime.

We provide the file **check_unweighted.py** and **check_weighted.py**, based on networkx, which can be used to check the results printed by DAWN.

We also provide the test code for Gunrock and GAPBS in the **test** directory. Due to differences in code build environments and other aspects among the repositories, it is not possible to pull and build them uniformly. If you need to verify the results of Gunrock and GAPBS, please visit the repositories for [Gunrock](https://github.com/gunrock/gunrock) and [GAPBS](https://github.com/sbeamer/gapbs) respectively, follow the repository build instructions, and replace the source files in the repository with the ones we provide. Alternatively, you can pull our modified fork branch and build directly([Gunrock](https://github.com/lxrzlyr/gunrock),[GAPBS](https://github.com/lxrzlyr/gapbs)).

### 7.Documentation

Please refer to [document/Documentation_v1](https://github.com/lxrzlyr/SC2023/blob/eb9080f76c2950981a4dac72141d4991eff8b9db/document/Decumentation_v1.md) for commands.

## New version

The version of DWAN on the weighted graph has been included in DAWN V2.1. Currently, DAWN includes the version that runs on unweighted graphs of int type index values, and the version that runs on negative weighted graphs of float type. (SOVM and GOVM have been the default implementation, if you want to use BOVM, please change the kernel function.)

| Algorithm | Release |
| -------- | -------- |
| APSP |  V2.1 |
| MSSP |  V2.1 |
| SSSP |  V2.1 |
| BFS  |  V2.1 |
| BC   |Future |
| CC   |Future |
| Cluster Analysis |Future |
| Community Detection |Future |

In the future, we plan to develop more algorithms based on DAWN, including but not limited to Between Centrality, Closeness Centrality, etc. Further applications of these algorithms, such as community detection, clustering, and path planning, are also on our agenda.

We welcome any interest and ideas related to DAWN and its applications. If you are interested in DAWN algorithms and their applications, please feel free to share your thoughts via [email](<1289539524@qq.com>), and we will do our best to assist you in your research based on DAWN.

The DAWN component based on Gunrock may be released to the main/develop branch in the near future, so please stay tuned to the [Gunrock](https://github.com/gunrock/gunrock). We will release new features of DAWN and the application algorithms based on DAWN on this repository. If the algorithms are also needed by Gunrock, we will contribute them to the Gunrock repository later.

## Copyright & License

All source code are released under [Apache 2.0](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/blob/4266d98053678ce76e34be64477ac2364f0f4291/LICENSE).
