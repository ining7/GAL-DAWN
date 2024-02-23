/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/mssp.hxx>

float DAWN::MSSP_CPU::runSG(Graph::Graph_t& graph, std::string& output_path) {
  float elapsed_time = 0.0f;
  float time = 0.0f;
  int proEntry = 0;
  auto row = graph.rows;

#pragma omp parallel for
  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % row;
    if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
      ++proEntry;
      continue;
    }
    if (graph.weighted) {
      time = DAWN::SSSP_CPU::SSSPs(graph, i, output_path);
    } else {
      time = DAWN::BFS_CPU::BFSs(graph, i, output_path);
    }
#pragma omp critical
    {
      elapsed_time += time;
      ++proEntry;
    }
  }
  elapsed_time = elapsed_time / (graph.stream * 1000);
  return elapsed_time;
}

float DAWN::MSSP_CPU::runTG(Graph::Graph_t& graph, std::string& output_path) {
  float elapsed_time = 0.0f;
  int proEntry = 0;
  auto row = graph.rows;

  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % row;
    if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
      ++proEntry;
      continue;
    }
    if (graph.weighted) {
      elapsed_time += DAWN::SSSP_CPU::SSSPp(graph, i, output_path);
    } else {
      elapsed_time += DAWN::BFS_CPU::BFSp(graph, i, output_path);
    }
    ++proEntry;
  }
  elapsed_time = elapsed_time / 1000;
  return elapsed_time;
}
