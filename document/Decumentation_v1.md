# 1.Function

```c++
    struct Matrix
    {
        int rows;
        int cols;
        uint64_t nnz;
        int **A;      // 按列压缩
        int *A_entry; // 每列项数
        int **B;      // 按行压缩
        int *B_entry; // 每行项数
        int dim;
        uint64_t entry;
        int thread;
        int interval;
        int stream;
        int block_size;
        bool prinft; // 是否打印结果
        int source;  // 打印的节点
    };
```

Matrix struct

````c++
    void runApspV3(Matrix &matrix, string &output_path);

    void runApspV4(Matrix &matrix, string &output_path);

    void runSsspCpu(DAWN::Matrix &matrix, std::string &output_path);

    void runApspGpu(DAWN::Matrix &matrix, std::string &output_path);

    void runSsspGpu(DAWN::Matrix &matrix, std::string &output_path);
````

Function of running dawn, time complexity O(dnm) and space complexity O(n^2)

````c++
    void createGraph(string &input_path, Matrix &matrix);

    void readGraph(string &input_path, Matrix &matrix, vector<pair<int, int>> &cooMatCol);

    void readGraphBig(string &input_path, string &col_input_path, string &row_input_path, Matrix &matrix);
````

Function of reading and creating graph from mtx file


# 2 Command

## 2.1 dawn_cpu_v1

````c++
    ./dawn_cpu_v1 -algo $GRAPH_DIR/xx.mtx ../outpu.txt -interval -prinft -source
    
    ./dawn_cpu_v1 -algo $GRAPH_DIR/xx.mtx $GRAPH_DIR/graph_CRC.txt $GRAPH_DIR/graph_RCC.txt ../outpu.txt -interval -prinft -source
````

**algo** is the algorithm, **FG** is fine-grained parallel version and **CG** is the coarse-grained parallel version.

**BFG** and **BCG** run on the large-scale graph.

**interval** is the refers to the progress interval. For example, **100** refers to printing the current time every time 1% of the node calculation is completed.

**prinft** refers to printing the calculation result, and **source** refers to the node that prints the result. If printing is not required, **prinft** is **false**, and **source** is filled in casually. If printing is required, **prinft** is **true**, and **source** is filled with the node number to be printed.

## 2.2 dawn_gpu_v1

````c++
    ./dawn_gpu_v1 -algo $GRAPH_DIR ../outpu.txt -stream -block_size -interval -prinft -source
    ./dawn_gpu_v1 -algo $GRAPH_DIR/xx.mtx $GRAPH_DIR/graph_CRC.txt $GRAPH_DIR/graph_RCC.txt ../outpu.txt -stream -block_size -interval -prinft -source
````

**algo** is the algorithm, where **Default** is the default version and **Big** is the version for large-scale graph.

**stream** is the CUDAstreams and **block_size** is the block_size on GPU, which are adjusted according to GPU hardware resources.
