/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/gpu/sssp.cuh>

int main(int argc, char* argv[]) {
  DAWN::IO::parameters_t params = DAWN::IO::parameters(argc, argv);
  DAWN::Graph::Graph_t graph;
  graph.source = params.source;
  graph.print = params.print;
  graph.weighted = true;
  graph.thread = 1;
  graph.stream = 1;
  graph.block_size = 1024;

  DAWN::Graph::createGraph(params.input_path, graph);
  float elapsed_time = DAWN::SSSP_GPU::run(graph, params.output_path);
  printf("%-21s%3.5d\n", "Nodes:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time);

  return 0;
}
//./sssp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 1024 false 0