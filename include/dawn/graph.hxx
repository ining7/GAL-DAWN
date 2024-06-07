/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-04-21
 *
 * @copyright Copyright (c) 2024
 */
#pragma once
#include <dawn/include.hxx>

namespace DAWN {
namespace Graph {

struct Csr_t {
 public:
  int* row_ptr;
  int* col;
  float* val;
};

struct Coo_t {
 public:
  int* row;
  int* col;
  float* val;
};

class Graph_t {
 public:
  int rows;
  int cols;
  uint64_t nnz;
  Csr_t csr;
  Coo_t coo;
  int thread;
  int interval;
  int stream;
  int block_size;
  bool print;  // print the result
  int source;
  bool weighted;
  bool directed;
  std::vector<int> msource;  // Source list for Multi-Source algorithms
};

void coo2Csr(int n,
             int nnz,
             Csr_t& csr,
             Coo_t& coo);  // COO matrix to CSR matrix

void coo2Csr_Weighted(int n, int nnz, Csr_t& csr, Coo_t& coo);

void transpose(int nnz, Coo_t& coo);

void transpose_Weighted(int nnz, Coo_t& coo);

void createGraph(std::string& input_path, Graph_t& graph);  // create graph

void readGraph(
    std::string& input_path,
    Graph_t& graph);  // read undirected and unweighted graph from file

void readGraph_Weighted(
    std::string& input_path,
    Graph_t& graph);  // read undirected and weighted graph from file

void readGraph_Directed(
    std::string& input_path,
    Graph_t& graph);  // read directed and unweighted graph from file

void readGraph_Directed_Weighted(
    std::string& input_path,
    Graph_t& graph);  // read directed and weighted graph from file

void readList(std::string& input_path,
              Graph_t& graph);  // read list from file

}  // namespace Graph
}  // namespace DAWN