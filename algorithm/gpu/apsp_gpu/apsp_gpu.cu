/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/gpu/apsp.cuh>
int main(int argc, char* argv[]) {
  DAWN::IO::parameters_t params = DAWN::IO::parameters(argc, argv);
  DAWN::Graph::Graph_t graph;
  graph.source = params.source;
  graph.print = params.print;
  graph.weighted = params.weighted;
  graph.thread = 1;
  graph.stream = 4;
  graph.block_size = 1024;
  graph.interval = 100;

  DAWN::Graph::createGraph(params.input_path, graph);

  if (graph.weighted) {
    float elapsed_time =
        DAWN::APSP_GPU::run_Weighted(graph, params.output_path);
    printf("%-21s%3.5d\n", "Nodes:", graph.rows);
    printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
    printf("%-21s%3.5lf\n", "Time:", elapsed_time);
  } else {
    float elapsed_time = DAWN::APSP_GPU::run(graph, params.output_path);
    printf("%-21s%3.5d\n", "Nodes:", graph.rows);
    printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
    printf("%-21s%3.5lf\n", "Time:", elapsed_time);
  }

  return 0;
}