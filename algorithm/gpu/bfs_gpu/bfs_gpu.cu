#include <dawn/dawn.cuh>

int main(int argc, char* argv[]) {
  DAWN::Graph graph;
  DAWN::GPU gpurun;

  std::string input_path = argv[1];
  std::string output_path = argv[2];
  graph.block_size = atoi(argv[3]);
  std::string prinft = argv[4];
  graph.source = atoi(argv[5]);

  graph.thread = 1;
  graph.stream = 1;
  graph.weighted = false;

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }

  graph.createGraph(input_path, graph);
  gpurun.runBFSGpu(graph, output_path);

  return 0;
}
//./sssp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 1024 false 0