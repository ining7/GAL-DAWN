/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/cc.hxx>
float DAWN::CC_CPU::Closeness_Centrality(Graph::Graph_t& graph, int source) {
  int step = 1;
  bool is_converged = false;
  auto row = graph.rows;
  bool* alpha = new bool[row];
  bool* beta = new bool[row];
  int* distance = new int[row];
  float elapsed = 0.0f;

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, 0);

#pragma omp parallel for
  for (int i = graph.csr.row_ptr[source]; i < graph.csr.row_ptr[source + 1];
       i++) {
    alpha[graph.csr.col[i]] = true;
    distance[graph.csr.col[i]] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;

    if (!(step % 2))
      if (graph.weighted)
        is_converged = DAWN::BFS_CPU::SOVMP(graph, alpha, beta, distance, step);
      else
        is_converged = DAWN::BFS_CPU::SOVMP(graph, beta, alpha, distance, step);
    if (is_converged) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  distance[source] = 0;
  float closeness_centrality =
      (row - 1) / std::accumulate(distance, distance + row, 0);

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;

  printf("%-21s%3.5d\n", "Node:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed / (graph.thread * 1000));

  return closeness_centrality;
}

float DAWN::CC_CPU::Closeness_Centrality_Weighted(Graph::Graph_t& graph,
                                                  int source) {
  int step = 1;
  bool is_converged = false;
  auto row = graph.rows;
  bool* alpha = new bool[row];
  bool* beta = new bool[row];
  float* distance = new float[row];
  float elapsed = 0.0f;
  float INF = 1.0 * 0xfffffff;

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, INF);

#pragma omp parallel for
  for (int i = graph.csr.row_ptr[source]; i < graph.csr.row_ptr[source + 1];
       i++) {
    distance[graph.csr.col[i]] = graph.csr.val[i];
    alpha[graph.csr.col[i]] = true;
  }
  distance[source] = 0.0f;

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      is_converged = DAWN::SSSP_CPU::GOVMP(graph, alpha, beta, distance);
    else
      is_converged = DAWN::SSSP_CPU::GOVMP(graph, beta, alpha, distance);
    if (is_converged) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  distance[source] = 0;
  float closeness_centrality =
      (1.0 * row - 1.0f) / std::accumulate(distance, distance + row, 0.0f);

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;

  printf("%-21s%3.5d\n", "Node:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed / (graph.thread * 1000));

  return closeness_centrality;
}
