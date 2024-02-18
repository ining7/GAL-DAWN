#include <dawn/dawn.hxx>

int main(int argc, char* argv[]) {
  DAWN::CPU runCpu;
  DAWN::Graph graph;
  std::string input_path = argv[1];
  std::string output_path = argv[2];
  std::string prinft = argv[3];
  graph.source = atoi(argv[4]);
  std::string weighted = argv[5];

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }

  graph.stream = 1;
  graph.thread = omp_get_num_threads();

  if (weighted == "true") {
    graph.weighted = true;
    graph.createGraph(input_path, graph);
    runCpu.Betweenness_Centrality_Weighted(graph, graph.source, output_path);
  } else {
    graph.weighted = false;
    graph.createGraph(input_path, graph);
    runCpu.Betweenness_Centrality(graph, graph.source, output_path);
  }

  return 0;
}
// ./bc_cpu $GRAPH_DIR/XX.mtx ./outpur.txt 0 weighted
