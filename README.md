# DAWN

## 0. Before getting started

Depending on your GPU, you may also want to edit the CMAKE_CUDA_ARCHITECTURES variable in $PROJECT_ROOT/CMakeLists.txt

```c++
export PROJECT_ROOT="to_your_project_path"
```

## 1. Modify $PROJECT_ROOT/CMakeLists.txt

According to your GPU, we use RTX 3080ti for computing, so CMAKE_CUDA_ARCHITECTURES is set to 86

```c++
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_xx,code=sm_xx")
```

Set the CMAKE_CUDA_COMPILER to the path of your NVCC, for example,

```c++
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
```

## 2.Download testing data

Unzip the compressed package and put it in the directory you need

The input data can be found on the Science Data Bank

```c++
URL=https://www.scidb.cn/s/6BjM3a
GRAPH_DIR="to_your_graph_path"
```

## 3. Build and RUN

```c++
cd $PROJECT_ROOT
mkdir build && cd build
cmake .. && make -j
```

If compilation succeeds without errors, you can run your code as before, for example

```c++
cd $PROJECT_ROOT/build
./dawn_cpu_v1 SG $GRAPH_DIR/mouse_gene.mtx ../output.txt 100 false 0

./dawn_gpu_v1 Default $GRAPH_DIR/mouse_gene.mtx ../output.txt 4 256 100 false 0

```

When the version is built, it will also prepare SSSP applications, which can be used directly.

If you need to use DAWN in your own solution, please check the source code under the **sssp** folder and call it.

If you do not have the conditions to use NVCC, you need to comment out all GPU-related statements in **cmakelists.txt**, then use GCC or clang to build applications that can only run on the cpu. (GCC 9.4.0 and above, clang 10.0.0 and above)

## 4.Using script

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

## 5.Release version

| Version | Implementation |
| ------ | ------ |
| TG |  Version of Thread Parallel|
| SG |  Version of Stream Parallel|
|Mssp | Version of Multi-source based on Stream Parallel|

Currently, on the Single-Processor, SG allocates one thread per stream. On Multi-Processor, SG will allocate multiple threads per stream. (It will be supported in the next version.)

For the GPU version, you can use Default, please make sure that there is enough GPU memory for the graph.

We provide a faster GPU kernel function which uses a lot of shared memory. Compared with the Function **vecMatOpeCsr**, the **vecMatOpeCsrShare** will achieve a speedup above 1.4&times;. However, due to large differences in hardware parameters such as shared memory in various devices, we do not use this function in the **Default** mode. You can check the maximum share memory per block as foolows,

```c++
int device;
cudaDeviceProp props;
cudaGetDevice(&device);
cudaGetDeviceProperties(&props, device);
printf("Max shared memory per block: %ld\n", props.sharedMemPerBlock);
```

If you are sure that the size of the thread block and the scale of the graph is set reasonably according to the device parameters. If the requested shared memory exceeds the maximum available shared memory, the **vecMatOpeCsrShare** function will return an incorrect result.

Please modify the **graph.share** in the Default mode in the **dawn_gpu_v1.cu** file to **true**. If shared memory is sufficient, you can modify the **graph.stream** at the same time (the default vaule is 1). And then, you can test the fastest version of DAWN.

## 5.Release result

On the test machine with i5-13600KF, SG achieves average speedup of 4.535&times; and 4.178&times; over SSSP and BFS from Gunrock (hereinafter referred to as GDS and BFS), respectively. On the test machine with i5-13600KF, DAWN need more than 10GB free memory to solve the large graph [281K,214M].

On the 64-thread AMD EPYC 7T83, DAWN achieved speedups of 17.735&times; and 19.176&times;, over over SSSP and BFS from Gunrock, respectively.

On the test machine with RTX3080TI, DAWN achieves average speedup of 2.779&times; 4.534&times; and 4.173&times;, over SG, GDS and BFS, respectively.

We provide the file **check_by_networkx.py**, based on networkx, which can be used to check the results printed by DAWN.

## 6.New version

The version of DWAN on the weighted graph will be include in DAWN2.0, and we will start making new artifacts as soon as possible.

## 7.Decumentation

Please refer to [decument/Decumentation_v1](https://github.com/ining7/SC2023/blob/f37c968a6a7d2195587354fb7592261e70a4d2c8/document%C2%A0%E6%96%87%E6%A1%A3/Decumentation_v1.md) for commands.
