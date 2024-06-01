# 1. Introduction
We have presented a performance comparison of algorithms for DAWN, GAPBS, and Gunrock. The benchmark tests were run on the Gunrock benchmark dataset and the Suite Sparse Collection dataset. The table provides specific information about the graphs and their corresponding runtime. The baseline implementations from Gunrock and GAPBS are provided in the **test** directory.

# 2. Test Environment

The test environment is as follows:

- OS: Ubuntu 20.04.5 LTS
- CPU: Intel Core i5-13600KF
- GPU: NVIDIA GeForce RTX 3080 TI
- Memory: 32GB
- CUDA: 12.1

# 3. Test Dataset

We leverage a set of 66 general graphs sourced from the SuiteSparse Matrix Collection and the Gunrock benchmark datasets. It can be accessed via the following link:
```cpp
https://www.scidb.cn/s/6BjM3a
```

# 4. Test Code

We also provide the test code for Gunrock in the **test/gunrock** and GAPBS in the **test/gapbs**. Due to differences in code build environments and other aspects among the repositories, it is not possible to pull and build them uniformly. Alternatively, you can pull our modified fork branch and build directly([Gunrock](https://github.com/lxrzlyr/gunrock),[GAPBS](https://github.com/lxrzlyr/gapbs)).

If you need to verify the results of Gunrock and GAPBS, please visit the repositories for [Gunrock](https://github.com/gunrock/gunrock) and [GAPBS](https://github.com/sbeamer/gapbs) respectively, follow the repository build instructions, and replace the source files in the repository with the ones we provide.

# 5. Test Procedure
Please check out the [Quick Start](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/blob/dev/document/Quick_Start.md) for the procedure.

```c++
cd $PROJECT_ROOT
mkdir build && cd build
cmake .. && make -j
```
If compilation succeeds without errors, revise script **test_process.sh**.

```c++
MAIN = ${PROJECT_ROOT}/build
GRAPH_DIR = ${test_graph}
SourceList = ${SourceList}
LOG_DIR= ${GRAPH_DIR}/log
```

Please note that the normal operation of the batch script needs to ensure that the test machine meets the minimum requirements. Insufficient memory or GPU memory needs to be manually adjusted according to amount of resources. For the GPU version, please make sure that there is enough GPU memory for the graph. 

All parameters have been set. If you are not familiar with them, please refrain from making casual modifications. Any change in any parameter may result in differences between the test results and the performance reference data we have provided.

# 6. [Performance Reference Data](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/tree/dev/test/Performance.md) 

# 7. Check the Results
We provide the file **check_unweighted.py** and **check_weighted.py**, based on networkx, which can be used to check the results printed by DAWN. The usage of the networkx can be find in [networkx](https://networkx.org/).
