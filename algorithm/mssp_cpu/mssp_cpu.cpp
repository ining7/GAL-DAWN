#include <dawn/dawn.hxx>

int main(int argc, char* argv[])
{
  DAWN::Tool  tool;
  DAWN::CPU   runCpu;
  DAWN::Graph graph;
  std::string input_path  = argv[1];
  std::string output_path = argv[2];
  std::string prinft      = argv[3];
  std::string sourceList  = argv[4];
  std::string weighted    = argv[5];
  graph.interval          = 100;
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
  graph.stream = 20;
  graph.thread = 20;
  graph.createGraph(input_path, graph);
  graph.readList(sourceList, graph);
  runCpu.runMsspP(graph, output_path);

  return 0;
}
// ./dawn_cpu_mssp $GRAPH_DIR/XX.mtx ../output.txt false sourceList unweighted//
// ./dawn_cpu_mssp $GRAPH_DIR/XX.mtx ../output.txt false sourceList weighted
