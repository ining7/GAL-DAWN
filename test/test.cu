#include "dawn.cuh"
int main(int argc, char* argv[]) {
  DAWN::Graph graph;
  DAWN::GPU gpurun;

  std::string input_path = argv[1];
  std::string output_path = argv[2];
  std::string prinft = argv[3];
  graph.source = atoi(argv[4]);
  std::string weighted = argv[5];

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else
    graph.prinft = false;

  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  graph.stream = 4;
  graph.block_size = 1024;
  graph.interval = 100;
  graph.createGraph(input_path, graph);
  if (weighted == "weighted") {
    graph.weighted = true;
    gpurun.runAPSPGpu(graph, output_path);
    // std::cout << "Weighted Graph" << std::endl;
  } else {
    graph.weighted = false;
    gpurun.runAPBFSGpu(graph, output_path);
  }

  return 0;
}