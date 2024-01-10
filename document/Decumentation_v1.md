# 1.Class

```c++
class Graph {
public:
  struct Csr
  {
    int*   row_ptr;
    int*   col;
    float* val;
  };
  struct Coo
  {
    int*   row;
    int*   col;
    float* val;
  };
  int              rows;
  int              cols;
  uint64_t         nnz;
  Csr              csrA;
  Csr              csrB;
  Coo              coo;
  uint64_t         entry;
  int              thread;
  int              interval;
  int              stream;
  int              block_size;
  bool             prinft;  // prinft the result
  int              source;
  bool             share;
  bool             weighted;
  bool             directed;
  std::vector<int> msource;
  float            MAX = (float)pow(2, 30);

  void createGraph(std::string& input_path, Graph& graph);

  void readGraph(std::string& input_path, Graph& graph);

  void readGraphW(std::string& input_path, Graph& graph);

  void readGraphD(std::string& input_path, DAWN::Graph& graph);

  void readGraphDW(std::string& input_path, DAWN::Graph& graph);

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
  float BFSPCsr(Graph& graph, int source, std::string& output_path);

  float ssspSCsr(Graph& graph, int source, std::string& output_path);

};
````

````c++
  class Tool {
public:
  void coo2Csr(int n, int nnz, Graph::Csr& csr, Graph::Coo& coo);

  void transpose(int n, int nnz, Graph::Coo& coo);

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

};
````

````c++
__global__ void vecMatOpeCsr(bool* input,
                             bool* output,
                             int*  result,
                             int*  rows,
                             int*  source,
                             int*  d_entry);
__global__ void vecMatOpeCsrShare(bool* input,
                                  bool* output,
                                  int*  result,
                                  int*  rows,
                                  int*  source,
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
