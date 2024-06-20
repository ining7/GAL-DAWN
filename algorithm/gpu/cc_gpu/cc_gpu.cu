/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/gpu/cc.cuh>

int main(int argc, char* argv[]) {
  DAWN::IO::parameters_t params = DAWN::IO::parameters(argc, argv);
  DAWN::Graph::Graph_t graph;
  graph.source = params.source;
  graph.print = params.print;
  graph.weighted = params.weighted;
  graph.thread = 1;
  graph.stream = 1;
  graph.block_size = 1024;

  DAWN::Graph::createGraph(params.input_path, graph);

  if (graph.weighted) {
    float closeness_centrality =
        DAWN::CC_GPU::run_Weighted(graph, graph.source);
    printf("%-21s%3.5d\n", "Source:", graph.source);
    printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);
  } else {
    float closeness_centrality = DAWN::CC_GPU::run(graph, graph.source);
    printf("%-21s%3.5d\n", "Source:", graph.source);
    printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);
  }

  return 0;
}
//./sssp_gpu $GRAPH_DIR/XXX.mtx 1024 0 false