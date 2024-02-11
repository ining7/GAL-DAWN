#include <dawn/dawn.cuh>

int main(int argc, char* argv[]) {
  DAWN::Graph graph;
  DAWN::GPU gpurun;
  std::string input_path = argv[1];
  std::string output_path = argv[2];
  graph.stream = atoi(argv[3]);
  graph.block_size = atoi(argv[4]);
  std::string prinft = argv[5];
  std::string sourceList = argv[6];
  std::string weighted = argv[7];

  graph.interval = 100;

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }

  graph.createGraph(input_path, graph);
  graph.readList(sourceList, graph);

  if (weighted == "weighted") {
    graph.weighted = true;
    gpurun.runMSSPGpu(graph, output_path);
    // std::cout << "Weighted Graph" << std::endl;
  } else {
    graph.weighted = false;
    gpurun.runMBFSGpu(graph, output_path);
  }

  return 0;
}

//./mssp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 8 1024 false ./sourceList.txt
// unweighted
//./mssp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 8 1024 false ./sourceList.txt
// weighted