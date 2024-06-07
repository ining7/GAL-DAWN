/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/gpu/bfs.cuh>

int main(int argc, char* argv[]) {
  DAWN::Graph::Graph_t graph;

  std::string input_path = argv[1];
  std::string output_path = argv[2];
  graph.block_size = atoi(argv[3]);
  std::string print = argv[4];
  graph.source = atoi(argv[5]);

  graph.thread = 1;
  graph.stream = 1;
  graph.weighted = false;

  if (print == "true") {
    graph.print = true;
    std::cout << "Print source " << graph.source << std::endl;
  } else {
    graph.print = false;
  }

  DAWN::Graph::createGraph(input_path, graph);

  float elapsed_time = DAWN::BFS_GPU::run(graph, output_path);

  printf("%-21s%3.5d\n", "Nodes:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time);

  return 0;
}
//./bfs_gpu $GRAPH_DIR/XXX.mtx ../output.txt 1024 false 0