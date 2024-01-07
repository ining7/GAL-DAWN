#include <dawn/dawn.cuh>
int main(int argc, char* argv[])
{
  DAWN::Graph graph;
  DAWN::GPU   gpurun;
  std::string algo        = argv[1];
  std::string input_path  = argv[2];
  std::string output_path = argv[3];
  graph.stream            = atoi(argv[4]);
  graph.block_size        = atoi(argv[5]);
  graph.interval          = atoi(argv[6]);
  std::string prinft      = argv[7];
  graph.source            = atoi(argv[8]);
  std::string weighted    = argv[9];

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
  graph.createGraph(input_path, graph);
  gpurun.runApspGpu(graph, output_path);

  return 0;
}
//./dawn_gpu_apsp Default $GRAPH_DIR/XXX.mtx ../output.txt 4 256 100
// false 0 unweighted
//./dawn_gpu_apsp Test $GRAPH_DIR/XXX.mtx ../output.txt
// false 0 unweighted
//./dawn_gpu_mssp Mssp $GRAPH_DIR/XXX.mtx ../output.txt 4 256 100
// false sourceList unweighted