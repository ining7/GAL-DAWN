#include <dawn/dawn.hxx>

int main(int argc, char* argv[]) {
  DAWN::CPU runCpu;
  DAWN::Graph graph;
  std::string input_path = argv[1];
  graph.source = atoi(argv[2]);
  std::string weighted = argv[3];

  graph.stream = 1;
  graph.thread = omp_get_num_threads();
  graph.createGraph(input_path, graph);

  if (weighted == "true") {
    graph.weighted = true;
    std::cout << "Prinft source " << graph.source << std::endl;
    runCpu.Closeness_Centrality_Weighted(graph, graph.source);
  } else {
    graph.weighted = false;
    std::cout << "Prinft source " << graph.source << std::endl;
    runCpu.Closeness_Centrality(graph, graph.source);
  }

  return 0;
}
// ./cc_cpu $GRAPH_DIR/XX.mtx 0 weighted
