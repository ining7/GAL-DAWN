#include "dawn.hxx"

int main(int argc, char* argv[])
{
  DAWN::Tool  tool;
  DAWN::CPU   runCpu;
  DAWN::Graph graph;
  std::string input_path  = argv[1];
  std::string output_path = argv[2];
  std::string prinft      = argv[3];
  graph.source            = atoi(argv[4]);
  std::string sourceList  = argv[5];
  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }

  graph.stream = 20;
  graph.thread = 20;
  graph.createGraphCsr(input_path, graph);
  graph.readList(sourceList, graph);
  runCpu.runMsspCpuCsr(graph, output_path);

  return 0;
}