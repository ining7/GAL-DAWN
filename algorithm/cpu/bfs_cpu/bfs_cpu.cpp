#include <dawn/dawn.hxx>

int main(int argc, char* argv[])
{
  DAWN::Tool  tool;
  DAWN::CPU   runCpu;
  DAWN::Graph graph;
  std::string input_path  = argv[1];
  std::string output_path = argv[2];
  std::string prinft      = argv[3];
  graph.source            = atoi(argv[4]);

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }
  graph.stream = 1;
  graph.thread = omp_get_num_threads();
  graph.createGraph(input_path, graph);
  runCpu.runBFS(graph, output_path);

  return 0;
}
// ./dawn_cpu_sssp $GRAPH_DIR/XX.mtx ../output.txt false 0 unweighted//
// ./dawn_cpu_sssp $GRAPH_DIR/XX.mtx ../output.txt false 0 weighted