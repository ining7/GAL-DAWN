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
      graph.stream         = atoi(argv[4]);
      graph.block_size     = atoi(argv[5]);
      graph.interval       = atoi(argv[6]);
      std::string prinft   = argv[7];
      graph.source         = atoi(argv[8]);
      std::string weighted = argv[9];

      if (prinft == "true") {
        graph.prinft = true;
        std::cout << "Prinft source " << graph.source << std::endl;
      } else {
        graph.prinft = false;
      }

      if (weighted == "weighted") {
        graph.weighted = true;
        std::cout << "Weighted Graph" << std::endl;
      } else {
        graph.weighted = false;
      }
      graph.createGraphGPU(input_path, graph);
      gpurun.runSsspGpu(graph, output_path);
    }

    if (algo == "Test") {
      graph.block_size     = atoi(argv[4]);
      std::string prinft   = argv[5];
      graph.source         = atoi(argv[6]);
      std::string weighted = argv[7];
      if (prinft == "true") {
        graph.prinft = true;
        std::cout << "Prinft source " << graph.source << std::endl;
      } else {
        graph.prinft = false;
      }
      graph.thread   = 1;
      graph.interval = 100;
      graph.share    = false;
      if (graph.share) {
        graph.stream = 16;
      } else {
        graph.stream = 16;
      }
      if (weighted == "weighted") {
        graph.weighted = true;
        std::cout << "Weighted Graph" << std::endl;
      } else {
        graph.weighted = false;
      }
      graph.createGraphGPU(input_path, graph);
      gpurun.runSsspGpu(graph, output_path);
    }
  }
  return 0;
}