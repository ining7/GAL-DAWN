#include "dawn.cuh"

int main(int argc, char* argv[])
{
  DAWN::Graph graph;
  DAWN::GPU   gpurun;
  std::string input_path  = argv[1];
  std::string output_path = argv[2];
  graph.block_size        = atoi(argv[3]);
  std::string prinft      = argv[4];
  std::string sourceList  = argv[5];
  std::string weighted    = argv[6];
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
  graph.stream   = 8;

  graph.createGraph(input_path, graph);
  graph.readList(sourceList, graph);
  gpurun.runMsspGpu(graph, output_path);
  return 0;
}

//./dawn_gpu_mssp $GRAPH_DIR/XXX.mtx ../output.txt 256
// false sourceList unweighted