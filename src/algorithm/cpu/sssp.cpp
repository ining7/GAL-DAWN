#include <dawn/algorithm/cpu/sssp.hxx>

float DAWN::SSSP_CPU::runSSSP(Graph::Graph_t& graph, std::string& output_path) {
  int source = graph.source;
  auto row = graph.rows;
  if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }
  float elapsed_time = SSSPp(graph, source, output_path) / 1000;
  return elapsed_time;
}

float DAWN::SSSP_CPU::SSSPp(Graph::Graph_t& graph,
                            int source,
                            std::string& output_path) {
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
      is_converged = GOVMP(graph, alpha, beta, distance);
    else
      is_converged = GOVMP(graph, beta, alpha, distance);
    if (is_converged) {
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

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed;
}

float DAWN::SSSP_CPU::SSSPs(Graph::Graph_t& graph,
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

int DAWN::SSSP_CPU::GOVM(Graph::Graph_t& graph,
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

bool DAWN::SSSP_CPU::GOVMP(Graph::Graph_t& graph,
                           bool*& alpha,
                           bool*& beta,
                           float*& distance) {
  bool converged = true;
  auto row = graph.rows;
#pragma omp parallel for
  for (int j = 0; j < row; j++) {
    if (alpha[j]) {
      int start = graph.csr.row_ptr[j];
      int end = graph.csr.row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          int index = graph.csr.col[k];
          float tmp = distance[j] + graph.csr.val[k];
          if (distance[index] > tmp) {
            distance[index] = std::min(distance[index], tmp);
            beta[index] = true;
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
