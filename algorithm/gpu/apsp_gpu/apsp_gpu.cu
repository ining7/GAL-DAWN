/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/gpu/apsp.cuh>
int main(int argc, char* argv[]) {
  DAWN::Graph::Graph_t graph;

  std::string input_path = argv[1];
  std::string output_path = argv[2];
  graph.stream = atoi(argv[3]);
  graph.block_size = atoi(argv[4]);
  std::string prinft = argv[5];
  graph.source = atoi(argv[6]);
  std::string weighted = argv[7];

  graph.interval = 100;

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }

  DAWN::Graph::createGraph(input_path, graph);

  if (weighted == "weighted") {
    graph.weighted = true;

    float elapsed_time = DAWN::APSP_GPU::run_Weighted(graph, output_path);

    printf("%-21s%3.5d\n", "Nodes:", graph.rows);
    printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
    printf("%-21s%3.5lf\n", "Time:", elapsed_time);
  } else {
    graph.weighted = false;

    float elapsed_time = DAWN::APSP_GPU::run(graph, output_path);

    printf("%-21s%3.5d\n", "Nodes:", graph.rows);
    printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
    printf("%-21s%3.5lf\n", "Time:", elapsed_time);
  }

  return 0;
}
//./apsp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 8 1024 false 0 unweighted
//./apsp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 8 1024 false 0 weighted