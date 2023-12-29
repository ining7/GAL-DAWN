#include "dawn.hxx"

int main(int argc, char* argv[])
{
  DAWN::CPU   runCpu;
  DAWN::Graph graph;
  std::string algo = argv[1];
  if ((algo == "TG") || (algo == "SG")) {
    std::string input_path  = argv[2];
    std::string output_path = argv[3];
    graph.interval = atoi(argv[4]);  // 请保证打印间隔小于节点总数，建议10-1000
    std::string prinft = argv[5];
    graph.source       = atoi(argv[6]);
    if (prinft == "true") {
      graph.prinft = true;
      std::cout << "Prinft source " << graph.source << std::endl;
    } else
      graph.prinft = false;

    if (algo == "SG") {
      graph.thread = 20;
      graph.createGraph(input_path, graph);
      runCpu.runApspSG(graph, output_path);
      return 0;
    } else {
      graph.thread = 1;
      graph.createGraph(input_path, graph);
      runCpu.runApspTG(graph, output_path);
      return 0;
    }
  }
  return 0;
}