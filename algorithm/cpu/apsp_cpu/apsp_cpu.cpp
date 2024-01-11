#include <dawn/dawn.hxx>

int main(int argc, char* argv[])
{
  DAWN::CPU   runCpu;
  DAWN::Graph graph;
  std::string algo = argv[1];
  if ((algo == "TG") || (algo == "SG")) {
    std::string input_path  = argv[2];
    std::string output_path = argv[3];
    std::string prinft      = argv[4];
    graph.source            = atoi(argv[5]);
    std::string weighted    = argv[6];

    graph.interval = 100;

    if (prinft == "true") {
      graph.prinft = true;
      std::cout << "Prinft source " << graph.source << std::endl;
    } else
      graph.prinft = false;

    if (weighted == "weighted") {
      graph.weighted = true;
      // std::cout << "Weighted Graph" << std::endl;
    } else {
      graph.weighted = false;
    }

    if (algo == "SG") {
      graph.source = atoi(argv[6]);
      graph.stream = omp_get_num_threads();
      graph.thread = omp_get_num_threads();
      graph.createGraph(input_path, graph);
      runCpu.runAPSPSG(graph, output_path);

      return 0;
    }
    if (algo == "TG") {
      graph.source = atoi(argv[6]);
      graph.stream = 1;
      graph.thread = omp_get_num_threads();
      graph.createGraph(input_path, graph);
      runCpu.runAPSPTG(graph, output_path);
      return 0;
    }
  }
  return 0;
}

// ./apsp_cpu SG $GRAPH_DIR/XX.mtx ../output.txt false 0 unweighted//
// ./apsp_cpu SG $GRAPH_DIR/XX.mtx ../output.txt false 0 weighted