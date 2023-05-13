#include "dawn.cuh"

int main(int argc, char* argv[])
{
  DAWN::Graph graph;
  DAWN::GPU   gpurun;
  std::string algo        = argv[1];
  std::string input_path  = argv[2];
  std::string output_path = argv[3];
  graph.thread            = 1;
  if ((algo == "Default") || (algo == "Test")) {
    if (algo == "Default") {
      graph.stream       = atoi(argv[4]);
      graph.block_size   = atoi(argv[5]);
      graph.interval     = atoi(argv[6]);
      std::string prinft = argv[7];
      graph.source       = atoi(argv[8]);

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
      graph.createGraphCsr(input_path, graph);
      gpurun.runSsspGpuCsr(graph, output_path);
    }

    if (algo == "Test") {
      int         block_size = atoi(argv[4]);
      std::string prinft     = argv[5];
      graph.source           = atoi(argv[6]);
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
      graph.block_size = block_size;
      graph.interval   = 10;
      graph.stream     = 1;
      graph.createGraphCsr(input_path, graph);
      gpurun.runSsspGpuCsr(graph, output_path);
    }
  } else {
    std::cout << "Algorithm is illegal!" << std::endl;
  }
  return 0;
}