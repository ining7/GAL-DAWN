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
  Csr csrA;
  Csr csrB;
  Coo coo;
  uint64_t entry;
  int thread;
  int interval;
  int stream;
  int block_size;
  bool prinft;  // prinft the result
  int source;
  bool share;
  bool weighted;
  bool directed;
  std::vector<int> msource;

  void createGraph(std::string& input_path, Graph& graph);

  void readGraph(std::string& input_path, Graph& graph);

  void readGraphW(std::string& input_path, Graph& graph);

  void readGraphD(std::string& input_path, DAWN::Graph& graph);

  void readGraphDW(std::string& input_path, DAWN::Graph& graph);

  void readList(std::string& input_path, DAWN::Graph& graph);
};

class CPU {
 public:
  // APSP run
  void runAPSPTG(Graph& graph, std::string& output_path);

  void runAPSPSG(Graph& graph, std::string& output_path);

  // MSSP run
  void runMSSPTG(Graph& graph, std::string& output_path);

  void runMSSPSG(Graph& graph, std::string& output_path);

  // SSSP run
  void runSSSP(Graph& graph, std::string& output_path);

  void runBFS(Graph& graph, std::string& output_path);

  // kernel
  float BFSp(Graph& graph, int source, std::string& output_path);

  float BFSs(Graph& graph, int source, std::string& output_path);

  float BFSs(Graph& graph,
             int source,
             std::string& output_path,
             std::vector<float>& averageLenth);  // example

  float SSSPs(Graph& graph, int source, std::string& output_path);

  float SSSPp(Graph& graph, int source, std::string& output_path);

  // SOVM
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
  void coo2Csr(int n, int nnz, Graph::Csr& csr, Graph::Coo& coo);

  void coo2CsrW(int n, int nnz, DAWN::Graph::Csr& csr, DAWN::Graph::Coo& coo);

  void transpose(int nnz, Graph::Coo& coo);

  void transposeW(int nnz, DAWN::Graph::Coo& coo);

  float averageShortestPath(int* result, int n);

  float averageShortestPath(float* result, int n);

  void infoprint(int entry,
                 int total,
                 int interval,
                 int thread,
                 float elapsed_time);

  void outfile(int n, int* result, int source, std::string& output_path);

  void outfile(int n, float* result, int source, std::string& output_path);
};

}  // namespace DAWN