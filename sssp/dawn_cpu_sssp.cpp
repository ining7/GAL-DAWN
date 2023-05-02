#include "dawn.hxx"

int main(int argc, char* argv[])
{
  DAWN::Tool  tool;
  DAWN::CPU   runCpu;
  DAWN::Graph graph;
  std::string algo = argv[1];
  if (algo == "Default") {
    std::string input_path  = argv[2];
    std::string output_path = argv[3];
    graph.interval = atoi(argv[4]);  // 请保证打印间隔小于节点总数，建议10-1000
    std::string prinft = argv[5];
    graph.source       = atoi(argv[6]);
    if (prinft == "true") {
      graph.prinft = true;
      std::cout << "Prinft source " << graph.source << std::endl;
    } else {
      graph.prinft = false;
    }
    graph.thread = 1;
    graph.createGraphCsr(input_path, graph);
    runCpu.runApspFGCsr(graph, output_path);
  } else {
    std::cout << "Algorithm is illegal!" << std::endl;
  }
  return 0;
}