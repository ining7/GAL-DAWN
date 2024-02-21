#include <dawn/algorithm/cpu/sssp.hxx>

int main(int argc, char* argv[]) {
  DAWN::Graph::Graph_t graph;
  std::string input_path = argv[1];
  std::string output_path = argv[2];
  std::string prinft = argv[3];
  graph.source = atoi(argv[4]);

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }
  graph.stream = 1;
  graph.thread = omp_get_num_threads();
  graph.weighted = true;

  DAWN::Graph::createGraph(input_path, graph);
  float elapsed_time = DAWN::SSSP_CPU::runSSSP(graph, output_path);
  printf("%-21s%3.5d\n", "Nodes:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time);

  return 0;
}
// ./sssp_cpu $GRAPH_DIR/XX.mtx ../output.txt false 0