/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/bfs.hxx>

float DAWN::BFS_CPU::run(Graph::Graph_t& graph, std::string& output_path) {
  int source = graph.source;
  auto row = graph.rows;
  if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }
  float elapsed_time = DAWN::BFS_CPU::BFSp(graph, source, output_path) / 1000;
  return elapsed_time;
}

// kernel
float DAWN::BFS_CPU::BFSp(Graph::Graph_t& graph,
                          int source,
                          std::string& output_path) {
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

  // Output
  if ((graph.prinft) && (source == graph.source)) {
    printf("Start prinft\n");
    DAWN::Tool::outfile(row, distance, source, output_path);
  }

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;
  return elapsed;
}

float DAWN::BFS_CPU::BFSs(Graph::Graph_t& graph,
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
      entry = DAWN::BFS_CPU::SOVM(graph, alpha, beta, distance, step, entry);
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

int DAWN::BFS_CPU::SOVM(Graph::Graph_t& graph,
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

bool DAWN::BFS_CPU::SOVMP(Graph::Graph_t& graph,
                          bool*& alpha,
                          bool*& beta,
                          int*& distance,
                          int step) {
  bool converged = true;
  auto row = graph.rows;
#pragma omp parallel for
  for (int j = 0; j < row; j++) {
    if (alpha[j]) {
      int start = graph.csr.row_ptr[j];
      int end = graph.csr.row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (!distance[graph.csr.col[k]]) {
            distance[graph.csr.col[k]] = step;
            beta[j] = true;
            if (converged)
              converged = false;
          }
        }
      }
      alpha[j] = false;
    }
  }
  return converged;
}