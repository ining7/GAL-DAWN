/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-04-21
 *
 * @copyright Copyright (c) 2024
 */
#pragma once
#include <dawn/include.hxx>
#include <dawn/matrix.hxx>

namespace DAWN {
namespace Graph {

class Graph_t {
 public:
  int rows;
  int cols;
  uint64_t nnz;
  DAWN::Matrix::Csr_t csr;
  DAWN::Matrix::Coo_t coo;
  int thread;
  int interval;
  int stream;
  int block_size;
  int source;
  bool print;
  bool weighted;
  bool directed;
  std::vector<int> msource;  // Source list for Multi-Source algorithms
};

void createGraph(std::string& input_path,
                 DAWN::Graph::Graph_t& graph);  // create graph

}  // namespace Graph
}  // namespace DAWN