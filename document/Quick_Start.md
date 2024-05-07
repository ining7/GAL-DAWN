# Quick Start Guide

## 0. Before getting started

If the machine does not have a GPU available, DAWN will automatically detect the CUDA environment and only build artifacts for the CPU.

```c++
export PROJECT_ROOT="to_your_project_path"
```
## 1. Modify $PROJECT_ROOT/algorithm/gpu/CMakeLists.txt

If you want to use the GPU version of DAWN, you need to modify the following code in $PROJECT_ROOT/algorithm/gpu/CMakeLists.txt. According to your GPU, we use RTX 3080TI for computing, so CUDA_ARCHITECTURES is set to 86.

```c++
set(CUDA_ARCHITECTURES "86")
```

Certainly, if you are unaware of the CUDA_ARCHITECTURES, we have implemented code to automatically select the CUDA_ARCHITECTURES. However, this may not necessarily be the most optimal choice.

If you are certain that the machine has a usable GPU and you have been unable to build artifacts for the GPU correctly, we suspect there may be an issue with the CUDA environment. Please further check for any path-related problems.

## 2. Download testing data

Unzip the compressed package and put it in the directory you need. The input data can be found on the Science Data Bank.

```c++
URL=https://www.scidb.cn/s/6BjM3a
GRAPH_DIR="to_your_graph_path"
```

## 3. Dependencies

DAWN builds, runs, and has been tested on Ubuntu/Rocky Linux. Even though DAWN may work on other linux systems, we have not tested correctness or performance. DAWN is not available on Windows and cannot achieve the same performance on WSL systems. Please beware.

At the minimum, DAWN depends on the following software:

```c++
- A modern C++ compiler compliant with the C++ 14 standard
- GCC (>= 9.4.0 or Clang >= 10.0.0)
- CMake (>= 3.10)
- libomp (>= 10.0)
```

If you need run DAWN on the GPU, expand:

```c++
- CUDA (>= 11.0)
- thrust (>= 2.0)
```

## 4. Build and RUN

```c++
cd $PROJECT_ROOT
mkdir build && cd build
cmake .. && make -j
```

If compilation succeeds without errors, you can run your code as before, for example

```c++
cd $PROJECT_ROOT/build
./sssp_cpu $GRAPH_DIR/XX.mtx ../output.txt false 0
./sssp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 1024 false 0
```

If you need to use DAWN in your own solution, please check the source code under the **src**，**include** folder and call them.

### 4.1 Using script

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

Please note that the normal operation of the batch script needs to ensure that the test machine meets the minimum requirements. Insufficient memory or GPU memory needs to be manually adjusted according to amount of resources. For the GPU version, please make sure that there is enough GPU memory for the graph. 

```c++
// For general graphs
CPU: Multi-threaded processor supporting OpenMP API
RAM: 8GB or more
GPU: 1GB or more
Compiler: NVCC of CUDA 11.0 above
OS:  Ubuntu 20.04 and above
```
