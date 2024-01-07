#include <dawn/access.h>
// #include <thrust/host_vector.h>
namespace DAWN {
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

class CPU {
public:
  // APSP run
  void runApspTG(Graph& graph, std::string& output_path);

  void runApspSG(Graph& graph, std::string& output_path);

  // MSSP run
  void runMsspS(Graph& graph, std::string& output_path);

  void runMsspP(Graph& graph, std::string& output_path);

  void runSssp(Graph& graph, std::string& output_path);

  // SSSP
  float ssspP(Graph& graph, int source, std::string& output_path);

  float ssspS(Graph& graph, int source, std::string& output_path);

  float ssspS(Graph&              graph,
              int                 source,
              std::string&        output_path,
              std::vector<float>& averageLenth);

  float ssspPW(Graph& graph, int source, std::string& output_path);

  float ssspSW(Graph& graph, int source, std::string& output_path);

  // BOVM
  void BOVM(Graph& graph,
            bool*& input,
            bool*& output,
            int*&  result,
            bool&  ptr,
            int    step);

  // SOVM
  void SOVMS(Graph& graph,
             int*&  alpha,
             int&   ptr,
             int*&  delta,
             int*&  result,
             int    step);

  void SOVMP(Graph& graph,
             bool*& alpha,
             int&   ptr,
             bool*& delta,
             int*&  result,
             int    step);
};

class Tool {
public:
  void coo2Csr(int n, int nnz, Graph::Csr& csr, Graph::Coo& coo);

  void coo2CsrW(int n, int nnz, DAWN::Graph::Csr& csr, DAWN::Graph::Coo& coo);

  void transpose(int nnz, Graph::Coo& coo);

  void transposeW(int nnz, DAWN::Graph::Coo& coo);

  float averageShortestPath(int* result, int n);

  float averageShortestPath(float* result, int n);

  void
  infoprint(int entry, int total, int interval, int thread, float elapsed_time);

  void outfile(int n, int* result, int source, std::string& output_path);

  void outfile(int n, float* result, int source, std::string& output_path);
};

}  // namespace DAWN