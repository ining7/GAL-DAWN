/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/cc.hxx>

int main(int argc, char* argv[]) {
  DAWN::IO::parameters_t params = DAWN::IO::parameters(argc, argv);
  DAWN::Graph::Graph_t graph;
  graph.source = params.source;
  graph.print = params.print;
  graph.weighted = params.weighted;

  graph.thread = omp_get_num_threads();
  DAWN::Graph::createGraph(params.input_path, graph);

  float elapsed_time = 0.0f;
  if (params.weighted) {
    float closeness_centrality =
        DAWN::CC_CPU::run_Weighted(graph, graph.source, elapsed_time);

    printf("%-21s%3.5d\n", "Source:", graph.source);
    printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);
    printf("%-21s%3.5d\n", "Node:", graph.rows);
    printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
    printf("%-21s%3.5lf\n", "Time:", elapsed_time);
  } else {
    float closeness_centrality =
        DAWN::CC_CPU::run(graph, graph.source, elapsed_time);

    printf("%-21s%3.5d\n", "Source:", graph.source);
    printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);
    printf("%-21s%3.5d\n", "Node:", graph.rows);
    printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
    printf("%-21s%3.5lf\n", "Time:", elapsed_time);
  }

  return 0;
}
