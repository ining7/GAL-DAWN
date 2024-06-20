/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/bfs.hxx>

int main(int argc, char* argv[]) {
  DAWN::IO::parameters_t params = DAWN::IO::parameters(argc, argv);
  DAWN::Graph::Graph_t graph;
  graph.source = params.source;
  graph.print = params.print;
  graph.weighted = false;
  graph.thread = omp_get_num_threads();

  DAWN::Graph::createGraph(params.input_path, graph);
  float elapsed_time = DAWN::BFS_CPU::run(graph, params.output_path);

  printf("%-21s%3.5d\n", "Nodes:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time);

  return 0;
}
