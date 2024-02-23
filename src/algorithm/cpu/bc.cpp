/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/bc.hxx>

float DAWN::BC_CPU::Betweenness_Centrality(DAWN::Graph::Graph_t& graph,
                                           std::string& output_path) {
  float elapsed_time = 0.0;
  auto row = graph.rows;
  for (int i = 0; i < row; i++) {
    if (graph.csr.row_ptr[i] == graph.csr.row_ptr[i + 1]) {
      DAWN::Tool::infoprint(i, row, graph.interval, graph.stream, elapsed_time);
      continue;
    }
    elapsed_time += DAWN::BC_CPU::kernel(graph, i, output_path);
    DAWN::Tool::infoprint(i, row, graph.interval, graph.stream, elapsed_time);
  }
  elapsed_time = elapsed_time / 1000;
  return elapsed_time;
}

float DAWN::BC_CPU::Betweenness_Centrality_Weighted(DAWN::Graph::Graph_t& graph,
                                                    std::string& output_path) {
  float elapsed_time = 0.0;
  auto row = graph.rows;
  for (int i = 0; i < row; i++) {
    if (graph.csr.row_ptr[i] == graph.csr.row_ptr[i + 1]) {
      DAWN::Tool::infoprint(i, row, graph.interval, graph.stream, elapsed_time);
      continue;
    }
    elapsed_time += DAWN::BC_CPU::kernel_Weighted(graph, i, output_path);
    DAWN::Tool::infoprint(i, row, graph.interval, graph.stream, elapsed_time);
  }
  elapsed_time = elapsed_time / 1000;
  return elapsed_time;
}

float DAWN::BC_CPU::kernel(DAWN::Graph::Graph_t& graph,
                           int source,
                           std::string& output_path) {
  int step = 1;
  int entry = graph.csr.row_ptr[source + 1] - graph.csr.row_ptr[source];
  auto row = graph.rows;
  int* alpha = new int[row];
  int* beta = new int[row];
  int* distance = new int[row];
  float elapsed = 0.0f;

  std::fill_n(distance, row, 0);
  std::fill_n(alpha, row, 0);
  std::fill_n(beta, row, 0);

  for (int i = graph.csr.row_ptr[source]; i < graph.csr.row_ptr[source + 1];
       i++) {
    distance[graph.csr.col[i]] = 1;
    alpha[i - graph.csr.row_ptr[source]] = graph.csr.col[i];
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      entry = DAWN::BC_CPU::SOVM(graph, alpha, beta, distance, step, entry);
    else
      entry = DAWN::BFS_CPU::SOVM(graph, beta, alpha, distance, step, entry);
    if (!entry) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  distance[source] = 0;

  // Output
  if ((graph.prinft) && (source == graph.source)) {
    printf("Start prinft\n");
    DAWN::Tool::outfile(row, distance, source, output_path);
  }

  delete[] alpha;
  alpha = nullptr;
  delete[] beta;
  beta = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed;
}

float DAWN::BC_CPU::kernel_Weighted(DAWN::Graph::Graph_t& graph,
                                    int source,
                                    std::string& output_path) {
  int step = 1;
  int entry = graph.csr.row_ptr[source + 1] - graph.csr.row_ptr[source];
  auto row = graph.rows;
  int* alpha = new int[row];
  int* beta = new int[row];
  float* distance = new float[row];
  float elapsed = 0.0f;
  float INF = 1.0 * 0xfffffff;

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, INF);

  for (int i = graph.csr.row_ptr[source]; i < graph.csr.row_ptr[source + 1];
       i++) {
    distance[graph.csr.col[i]] = graph.csr.val[i];
    alpha[i - graph.csr.row_ptr[source]] = graph.csr.col[i];
  }

  distance[source] = 0.0f;

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      entry = GOVM(graph, alpha, beta, distance, entry);
    else
      entry = GOVM(graph, beta, alpha, distance, entry);
    if (!entry) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  // Output
  if ((graph.prinft) && (source == graph.source)) {
    printf("Start prinft\n");
    DAWN::Tool::outfile(row, distance, source, output_path);
  }

  delete[] alpha;
  alpha = nullptr;
  delete[] beta;
  beta = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed;
}

int DAWN::BC_CPU::SOVM(Graph::Graph_t& graph,
                       int*& alpha,
                       int*& beta,
                       int*& distance,
                       int step,
                       int entry) {
  int tmpEntry = 0;
  for (int j = 0; j < entry; j++) {
    int start = graph.csr.row_ptr[alpha[j]];
    int end = graph.csr.row_ptr[alpha[j] + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (!distance[graph.csr.col[k]]) {
          distance[graph.csr.col[k]] = step;
          beta[tmpEntry] = graph.csr.col[k];
          ++tmpEntry;
        }
      }
    }
  }
  return tmpEntry;
}

int DAWN::BC_CPU::GOVM(DAWN::Graph::Graph_t& graph,
                       int*& alpha,
                       int*& beta,
                       float*& distance,
                       int entry) {
  int tmpEntry = 0;
  for (int j = 0; j < entry; j++) {
    int start = graph.csr.row_ptr[alpha[j]];
    int end = graph.csr.row_ptr[alpha[j] + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        int index = graph.csr.col[k];
        float tmp = distance[j] + graph.csr.val[k];
        if (distance[index] > tmp) {
          distance[index] = std::min(distance[index], tmp);
          beta[tmpEntry] = index;
          ++tmpEntry;
        }
      }
    }
  }
  return tmpEntry;
}
