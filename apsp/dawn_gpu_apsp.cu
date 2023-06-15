#include "dawn.cuh"

int main(int argc, char* argv[])
{
  DAWN::Graph graph;
  DAWN::GPU   gpurun;
  std::string algo        = argv[1];
  std::string input_path  = argv[2];
  std::string output_path = argv[3];
  graph.weighted          = true;
  if ((algo == "Default") || (algo == "Test") || (algo == "Mssp")) {
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
      gpurun.runApspGpu(graph, output_path);
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
      graph.stream   = 8;

      if (weighted == "weighted") {
        graph.weighted = true;
        std::cout << "Weighted Graph" << std::endl;
      } else {
        graph.weighted = false;
      }
      graph.createGraphGPU(input_path, graph);
      gpurun.runApspGpu(graph, output_path);
    }
    if (algo == "Mssp") {
      std::cout << "Mssp" << std::endl;
      graph.block_size       = atoi(argv[4]);
      std::string prinft     = argv[5];
      std::string sourceList = argv[6];
      std::string weighted   = argv[7];
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
      graph.thread   = 1;
      graph.interval = 100;
      graph.share    = false;
      graph.stream   = 4;

      graph.createGraphGPU(input_path, graph);
      graph.readList(sourceList, graph);
      gpurun.runMsspGpu(graph, output_path);
    }
  } else {
    std::cout << "Algorithm is illegal!" << std::endl;
  }

  return 0;
}