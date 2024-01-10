#include <dawn/dawn.cuh>
int main(int argc, char* argv[])
{
  DAWN::Graph graph;
  DAWN::GPU   gpurun;
  std::string input_path  = argv[1];
  std::string output_path = argv[2];
  graph.stream            = atoi(argv[3]);
  graph.block_size        = atoi(argv[4]);
  std::string prinft      = argv[5];
  graph.source            = atoi(argv[6]);
  std::string weighted    = argv[7];

  graph.interval = 100;

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }

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
//./apsp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 8 1024 false 0 unweighted
//./apsp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 8 1024 false 0 weighted