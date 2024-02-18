#include <dawn/access.h>

namespace DAWN {
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
  int thread;
  int interval;
  int stream;
  int block_size;
  bool prinft;  // prinft the result
  int source;
  bool weighted;
  bool directed;
  std::vector<int> msource;  // Source list for Multi-Source algorithms

  void createGraph(std::string& input_path, Graph& graph);  // create graph

  void readGraph(
      std::string& input_path,
      Graph& graph);  // read undirected and unweighted graph from file

  void readGraph_Weighted(
      std::string& input_path,
      Graph& graph);  // read undirected and weighted graph from file

  void readGraph_Directed(
      std::string& input_path,
      DAWN::Graph& graph);  // read directed and unweighted graph from file

  void readGraph_Directed_Weighted(
      std::string& input_path,
      DAWN::Graph& graph);  // read directed and weighted graph from file

  void readList(std::string& input_path,
                DAWN::Graph& graph);  // read list from file
};

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

class Tool {
 public:
  void coo2Csr(int n,
               int nnz,
               Graph::Csr& csr,
               Graph::Coo& coo);  // COO matrix to CSR matrix

  void coo2Csr_Weighted(int n,
                        int nnz,
                        DAWN::Graph::Csr& csr,
                        DAWN::Graph::Coo& coo);

  void transpose(int nnz, Graph::Coo& coo);

  void transpose_Weighted(int nnz, DAWN::Graph::Coo& coo);

  float average(int* result, int n);  // average value of a list

  float average(float* result, int n);

  void infoprint(int entry,
                 int total,
                 int interval,
                 int thread,
                 float elapsed_time);  // Print current task progress

  void outfile(int n, int* result, int source, std::string& output_path);

  void outfile(int n, float* result, int source, std::string& output_path);
};

}  // namespace DAWN