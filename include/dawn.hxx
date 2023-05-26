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
  std::vector<int> msource;

  void createGraphCsr(std::string& input_path, Graph& graph);

  void createGraphCsm(std::string& input_path, Graph& graph);

  void readGraph(std::string& input_path, Graph& graph);

  void readGraphWeighted(std::string& input_path, Graph& graph);

  void readList(std::string& input_path, DAWN::Graph& graph);
};

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

  // BOVM
  void BOVM(Graph& graph,
            bool*& output,
            bool*& input,
            int*&  result,
            int    dim,
            int&   entry);
  // SOVM
  void SOVM(Graph& graph,
            int*&  alpha,
            int&   alphaPtr,
            int*&  delta,
            bool*& beta,
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
};

}  // namespace DAWN