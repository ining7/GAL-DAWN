/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/cc.hxx>

int main(int argc, char* argv[]) {
  DAWN::Graph::Graph_t graph;
  std::string input_path = argv[1];
  graph.source = atoi(argv[2]);
  std::string weighted = argv[3];

  graph.stream = 1;
  graph.thread = omp_get_num_threads();
  DAWN::Graph::createGraph(input_path, graph);

  if (weighted == "true") {
    graph.weighted = true;

    float closeness_centrality =
        DAWN::CC_CPU::Closeness_Centrality_Weighted(graph, graph.source);

    printf("%-21s%3.5d\n", "Source:", graph.source);
    printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);
  } else {
    graph.weighted = false;

    float closeness_centrality =
        DAWN::CC_CPU::Closeness_Centrality(graph, graph.source);

    printf("%-21s%3.5d\n", "Source:", graph.source);
    printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);
  }

  return 0;
}
// ./cc_cpu $GRAPH_DIR/XX.mtx 0 weighted
