#include "dawn.cuh"

int main(int argc, char* argv[])
{
  DAWN::Graph graph;
  DAWN::GPU   gpurun;
  std::string input_path  = argv[1];
  std::string output_path = argv[2];
  graph.thread            = 1;
  graph.block_size        = atoi(argv[3]);
  std::string prinft      = argv[4];
  std::string sourceList  = argv[5];
  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return 0;
  }
  graph.thread   = 1;
  graph.interval = 100;
  graph.share    = false;
  if (graph.share) {
    graph.stream = 2;
  } else {
    graph.stream = 8;
  }
  graph.createGraph(input_path, graph);
  graph.readList(sourceList, graph);
  gpurun.runMsspGpu(graph, output_path);
  return 0;
}