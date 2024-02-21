#include <dawn/algorithm/cpu/apsp.hxx>

float DAWN::APSP_CPU::runAPSPTG(DAWN::Graph::Graph_t& graph,
                                std::string& output_path) {
  float elapsed_time = 0.0;
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  auto row = graph.rows;
  for (int i = 0; i < row; i++) {
    if (graph.csr.row_ptr[i] == graph.csr.row_ptr[i + 1]) {
      DAWN::Tool::infoprint(i, row, graph.interval, graph.stream, elapsed_time);
      continue;
    }
    if (graph.weighted) {
      elapsed_time += DAWN::SSSP_CPU::SSSPp(graph, i, output_path);
    } else {
      elapsed_time += DAWN::BFS_CPU::BFSp(graph, i, output_path);
    }
    DAWN::Tool::infoprint(i, row, graph.interval, graph.stream, elapsed_time);
  }
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;

  elapsed_time = elapsed_time / 1000;
  return elapsed_time;
}  // namespace

float DAWN::APSP_CPU::runAPSPSG(Graph::Graph_t& graph,
                                std::string& output_path) {
  float elapsed_time = 0.0;
  int proEntry = 0;
  float time = 0.0f;
  auto row = graph.rows;
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
#pragma omp parallel for
  for (int i = 0; i < row; i++) {
    if (graph.csr.row_ptr[i] == graph.csr.row_ptr[i + 1]) {
      ++proEntry;
      DAWN::Tool::infoprint(proEntry, row, graph.interval, graph.stream,
                            elapsed_time);
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
    DAWN::Tool::infoprint(proEntry, row, graph.interval, graph.stream,
                          elapsed_time);
  }
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  elapsed_time = elapsed_time / (graph.stream * 1000);
  return elapsed_time;
}
