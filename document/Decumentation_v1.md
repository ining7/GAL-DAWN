# 1.Class

```c++
class Graph {
 public:
  struct Csr {
    int* row_ptr;
    int* col;
    float* val;
  };
  struct Coo {
    int* row;
    int* col;
    float* val;
  };
  int rows;
  int cols;
  uint64_t nnz;
  Csr csr;
  Coo coo;
  uint64_t entry;
  int thread; //amount of threads
  int interval; //interval of print
  int stream; //amount of stream
  int block_size; 
  bool prinft;  
  int source;
  bool weighted;
  bool directed;
  std::vector<int> msource; // Source list for Multi-Source algorithms

  void createGraph(std::string& input_path, Graph& graph); // create graph

  void readGraph(std::string& input_path, Graph& graph);// read undirected and unweighted graph from file

  void readGraph_Weighted(std::string& input_path, Graph& graph);// read undirected and weighted graph from file

  void readGraph_Directed(std::string& input_path, DAWN::Graph& graph);// read directed and unweighted graph from file

  void readGraph_Directed_Weighted(std::string& input_path, DAWN::Graph& graph);// read directed and weighted graph from file

  void readList(std::string& input_path, DAWN::Graph& graph);// read list from file
};
```

````c++
   class CPU {
 public:
  // Shortest Path Algorithm
  void runAPSPTG(Graph& graph, std::string& output_path);

  void runAPSPSG(Graph& graph, std::string& output_path);

  void runMSSPTG(Graph& graph, std::string& output_path);

  void runMSSPSG(Graph& graph, std::string& output_path);

  void runSSSP(Graph& graph, std::string& output_path);

  void runBFS(Graph& graph, std::string& output_path);

  // Centrality Algorithm
  float Closeness_Centrality(Graph& graph, int source);

  float Closeness_Centrality_Weighted(Graph& graph, int source);

  float Betweenness_Centrality(Graph& graph,
                               int source,
                               std::string& output_path);

  float Betweenness_Centrality_Weighted(Graph& graph,
                                        int source,
                                        std::string& output_path);

  // kernel
  float BFSp(Graph& graph, int source, std::string& output_path);

  float BFSs(Graph& graph, int source, std::string& output_path);

  float SSSPs(Graph& graph, int source, std::string& output_path);

  float SSSPp(Graph& graph, int source, std::string& output_path);

  int SOVM(Graph& graph,
           int*& alpha,
           int*& beta,
           int*& distance,
           int step,
           int entry);

  int GOVM(Graph& graph, int*& alpha, int*& beta, float*& distance, int entry);

  bool SOVMP(Graph& graph, bool*& alpha, bool*& beta, int*& distance, int step);

  bool GOVMP(Graph& graph, bool*& alpha, bool*& beta, float*& distance);
};
````

````c++
class Tool {
 public:
  void coo2Csr(int n, int nnz, Graph::Csr& csr, Graph::Coo& coo);// COO matrix to CSR matrix

  void coo2Csr_Weighted(int n, int nnz, DAWN::Graph::Csr& csr, DAWN::Graph::Coo& coo);

  void transpose(int nnz, Graph::Coo& coo);

  void transpose_Weighted(int nnz, DAWN::Graph::Coo& coo);

  float average(int* result, int n);// average value of a list

  float average(float* result, int n);

  void infoprint(int entry,
                 int total,
                 int interval,
                 int thread,
                 float elapsed_time);// Print current task progress

  void outfile(int n, int* result, int source, std::string& output_path);// Output result to file

  void outfile(int n, float* result, int source, std::string& output_path);
};
````

````c++
class GPU {
 public:
  void runAPSPGpu(Graph& graph, std::string& output_path);

  void runAPBFSGpu(Graph& graph, std::string& output_path);

  void runMBFSGpu(Graph& graph, std::string& output_path);

  void runMSSPGpu(Graph& graph, std::string& output_path);

  void runSSSPGpu(Graph& graph, std::string& output_path);

  void runBFSGpu(Graph& graph, std::string& output_path);

  float BFSGpu(Graph& graph,
               int source,
               cudaStream_t streams,
               thrust::device_vector<int> d_row_ptr,
               thrust::device_vector<int> d_col,
               std::string& output_path);

  float SSSPGpu(Graph& graph,
                int source,
                cudaStream_t streams,
                thrust::device_vector<int> d_row_ptr,
                thrust::device_vector<int> d_col,
                thrust::device_vector<float> d_val,
                std::string& output_path);
};
````

# 2 Command

We have attached the command to call the function in a comment at the bottom of the source file.
