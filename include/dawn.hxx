#include "access.h"
#include <thrust/host_vector.h>
namespace DAWN {
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
    int*    row;  // CSR行指针
    int**   col;  // CSR列索引
    float** val;  // CSR值
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
  bool             weighted;
  std::vector<int> msource;
  float            MAX = (float)pow(2, 30);

  void createGraph(std::string& input_path, Graph& graph);

  void createGraphGPU(std::string& input_path, DAWN::Graph& graph);

  void createGraphCsm(std::string& input_path, Graph& graph);

  void readGraph(std::string& input_path, Graph& graph);

  void readGraphW(std::string& input_path, Graph& graph);

  void readList(std::string& input_path, DAWN::Graph& graph);
};

class CPU {
public:
  // APSP run
  void runApspTG(Graph& graph, std::string& output_path);

  void runApspSG(Graph& graph, std::string& output_path);

  // MSSP run
  void runMsspSCpu(Graph& graph, std::string& output_path);

  void runMsspPCpu(Graph& graph, std::string& output_path);

  void runSsspCpu(Graph& graph, std::string& output_path);

  // SSSP
  float ssspP(Graph& graph, int source, std::string& output_path);

  float ssspS(Graph& graph, int source, std::string& output_path);

  float ssspPW(Graph& graph, int source, std::string& output_path);

  float ssspSW(Graph& graph, int source, std::string& output_path);

  // APSP run
  void runApspTGCsm(Graph& graph, std::string& output_path);

  void runApspSGCsm(Graph& graph, std::string& output_path);

  // SSSP run
  void runSsspCpuCsm(Graph& graph, std::string& output_path);

  // SSSP
  float ssspPCsm(Graph& graph, int source, std::string& output_path);

  float ssspSCsm(Graph& graph, int source, std::string& output_path);

  // BOVM
  void BOVM(Graph& graph,
            bool*& input,
            bool*& output,
            int*&  result,
            int    dim,
            bool&  ptr);

  // SOVM
  void SOVMS(Graph& graph,
             int*&  alpha,
             int&   ptr,
             int*&  delta,
             int*&  result,
             int    dim);

  void SOVMP(Graph& graph,
             bool*& alpha,
             int&   ptr,
             bool*& delta,
             int*&  result,
             int    dim);
};

class Tool {
public:
  void coo2Csr(int n, int nnz, Graph::Csr& csr, Graph::Coo& coo);

  void csr2Csm(int n, int nnz, Graph::Csm& csm, Graph::Csr& csr);

  void coo2Csm(int n, int nnz, Graph::Csm& csm, Graph::Coo& coo);

  void transport(int n, int nnz, Graph::Coo& coo);

  void
  infoprint(int entry, int total, int interval, int thread, float elapsed_time);

  void outfile(int n, int* result, int source, std::string& output_path);

  void outfile(int                      n,
               thrust::host_vector<int> result,
               int                      source,
               std::string&             output_path);

  void outfile(int n, float* result, int source, std::string& output_path);
};

}  // namespace DAWN