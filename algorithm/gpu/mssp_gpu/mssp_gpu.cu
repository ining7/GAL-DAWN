/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/gpu/mssp.cuh>

int main(int argc, char* argv[]) {
  DAWN::IO::parameters_t params = DAWN::IO::parameters(argc, argv);
  DAWN::Graph::Graph_t graph;
  graph.print = params.print;
  graph.weighted = params.weighted;
  graph.source = params.source;
  graph.thread = 1;
  graph.stream = 4;
  graph.block_size = 1024;
  graph.interval = 100;

  DAWN::Graph::createGraph(params.input_path, graph);
  DAWN::IO::readList(params.sourceList_path, graph);

  if (graph.weighted) {
    float elapsed_time =
        DAWN::MSSP_GPU::run_Weighted(graph, params.output_path);
    printf("%-21s%3.5d\n", "Nodes:", graph.rows);
    printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
    printf("%-21s%3.5lf\n", "Time:", elapsed_time);
  } else {
    float elapsed_time = DAWN::MSSP_GPU::run(graph, params.output_path);
    printf("%-21s%3.5d\n", "Nodes:", graph.rows);
    printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
    printf("%-21s%3.5lf\n", "Time:", elapsed_time);
  }

  return 0;
}

//./mssp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 8 1024 false ./sourceList.txt
// unweighted
//./mssp_gpu $GRAPH_DIR/XXX.mtx ../output.txt 8 1024 false ./sourceList.txt
// weighted