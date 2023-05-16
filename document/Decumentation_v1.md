# 1.Class

```c++
   class Graph {
public:
  struct Csr         // 一维CSR
  {
    int*   row_ptr;  // CSR行指针
    int*   col;      // CSR列索引
    float* val;      // CSR值
  };
  struct Coo
  {
    int*   row;   // COO行指针
    int*   col;   // COO列索引
    float* val;   // COO值
  };
  struct Csm      // 二维CSR
  {
    int*    row;  // CSM行指针
    int**   col;  // CSM列索引
    float** val;  // CSM值
  };
  int              rows;
  int              cols;
  uint64_t         nnz;
  Csr              csrA;
  Csr              csrB;
  Coo              coo;
  Csm              csmA;
  Csm              csmB;
  int              dim;
  uint64_t         entry;
  int              thread;
  int              interval;
  int              stream;
  int              block_size;
  bool             prinft;  // 是否打印结果
  int              source;  // 打印的节点
  bool             share;
  std::vector<int> msource;

  void createGraphCsr(std::string& input_path, Graph& graph);

  void createGraphCsm(std::string& input_path, Graph& graph);

  void readGraph(std::string& input_path, Graph& graph);

  void readGraphWeighted(std::string& input_path, Graph& graph);

  void readList(std::string& input_path, DAWN::Graph& graph);
};
```

````c++
    class CPU {
public:
  // APSP run
  void runApspTGCsr(Graph& graph, std::string& output_path);

  void runApspSGCsr(Graph& graph, std::string& output_path);

  // SSSP run
  void runMsspCpuCsr(Graph& graph, std::string& output_path);

  void runSsspCpuCsr(Graph& graph, std::string& output_path);

  // SSSP
  float ssspPCsr(Graph& graph, int source, std::string& output_path);

  float ssspSCsr(Graph& graph, int source, std::string& output_path);

  // APSP run
  void runApspTGCsm(Graph& graph, std::string& output_path);

  void runApspSGCsm(Graph& graph, std::string& output_path);

  // SSSP run
  void runSsspCpuCsm(Graph& graph, std::string& output_path);

  // SSSP
  float ssspPCsm(Graph& graph, int source, std::string& output_path);

  float ssspSCsm(Graph& graph, int source, std::string& output_path);
};
````

````c++
  class Tool {
public:
  void coo2Csr(int n, int nnz, Graph::Csr& csr, Graph::Coo& coo);

  void csr2Csm(int n, int nnz, Graph::Csm& csm, Graph::Csr& csr);

  void coo2Csm(int n, int nnz, Graph::Csm& csm, Graph::Coo& coo);

  void transport(int n, int nnz, Graph::Coo& coo);

  void
  infoprint(int entry, int total, int interval, int thread, float elapsed_time);

  void outfile(int n, int* result, int source, std::string& output_path);
};
````

````c++
class GPU {
public:
  void runApspGpuCsr(Graph& graph, std::string& output_path);

  void runSsspGpuCsr(Graph& graph, std::string& output_path);

  void runMsspGpuCsr(Graph& graph, std::string& output_path);

  float ssspGpuCsr(Graph&       graph,
                   int          source,
                   cudaStream_t streams,
                   int*         d_A_row_ptr,
                   int*         d_A_col,
                   std::string& output_path);

  void runApspGpuCsm(Graph& graph, std::string& output_path);

  void runSsspGpuCsm(Graph& graph, std::string& output_path);

  float ssspGpuCsm(Graph&       graph,
                   int          source,
                   cudaStream_t streams,
                   int*         d_A_row_ptr,
                   int*         d_A_col,
                   std::string& output_path);
};
````

````c++
__global__ void vecMatOpeCsr(bool* input,
                             bool* output,
                             int*  result,
                             int*  rows,
                             int*  source,
                             int*  dim,
                             int*  d_entry);
__global__ void vecMatOpeCsrShare(bool* input,
                                  bool* output,
                                  int*  result,
                                  int*  rows,
                                  int*  source,
                                  int*  dim,
                                  int*  d_entry);
````

# 2 Command

## 2.1 dawn_cpu_v1

````c++
    ./dawn_cpu_v1 -algo $GRAPH_DIR/xx.mtx ../output.txt -interval -prinft -source
    ./dawn_cpu_v1 -algo $GRAPH_DIR/xx.mtx ../output.txt -interval -prinft -sourceList
````

**algo** is the algorithm, **TG** and **SG** is supported.

**interval** is the refers to the progress interval. For example, **100** refers to printing the current time every time 1% of the node calculation is completed.

**prinft** refers to printing the calculation result, and **source** refers to the node that prints the result. If printing is not required, **prinft** is **false**, and **source** is filled in casually. If printing is required, **prinft** is **true**, and **source** is filled with the node number to be printed.

## 2.2 dawn_gpu_v1

````c++
    ./dawn_gpu_v1 -algo $GRAPH_DIR ../output.txt -stream -block_size -interval -prinft -source
    ./dawn_gpu_v1 -algo $GRAPH_DIR ../output.txt -stream -block_size -interval -prinft -sourceList
````

**algo** is the algorithm, where **Default** is the default version and **Mssp** is the version for multi-source shortest path problem.

**stream** is the CUDAstreams and **block_size** is the block_size on GPU, which are adjusted according to GPU hardware resources.
