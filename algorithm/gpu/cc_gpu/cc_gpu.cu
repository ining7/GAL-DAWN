/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/gpu/cc.cuh>

int main(int argc, char* argv[]) {
  DAWN::Graph::Graph_t graph;

  std::string input_path = argv[1];
  graph.block_size = atoi(argv[2]);
  graph.source = atoi(argv[3]);
  std::string weighted = argv[4];

  graph.thread = 1;
  graph.stream = 1;

  DAWN::Graph::createGraph(input_path, graph);

  if (weighted == "true") {
    graph.weighted = true;

    float closeness_centrality =
        DAWN::CC_GPU::run_Weighted(graph, graph.source);

    printf("%-21s%3.5d\n", "Source:", graph.source);
    printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);
  } else {
    graph.weighted = false;

    float closeness_centrality = DAWN::CC_GPU::run(graph, graph.source);

    printf("%-21s%3.5d\n", "Source:", graph.source);
    printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);
  }

  return 0;
}
//./sssp_gpu $GRAPH_DIR/XXX.mtx 1024 0 false