#include "dawn.cuh"
int main(int argc, char* argv[])
{
  DAWN::Graph graph;
  DAWN::GPU   gpurun;
  std::string algo        = argv[1];
  std::string input_path  = argv[2];
  std::string output_path = argv[3];
  if (algo == "Test") {
    int         block_size = atoi(argv[4]);
    std::string prinft     = argv[5];
    graph.source           = atoi(argv[6]);
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
    graph.block_size = block_size;
    graph.thread     = 1;
    graph.interval   = 10;
    graph.stream     = 1;
    graph.createGraph(input_path, graph);
    gpurun.runApspGpu(graph, output_path);
  }
  return 0;
}