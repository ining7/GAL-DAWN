#include <dawn/dawn.hxx>
namespace DAWN {
void CPU::runAPSPTG(Graph& graph, std::string& output_path) {
  float elapsed_time = 0.0;
  Tool tool;
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  auto row = graph.rows;
  for (int i = 0; i < row; i++) {
    if (graph.csrB.row_ptr[i] == graph.csrB.row_ptr[i + 1]) {
      tool.infoprint(i, row, graph.interval, graph.stream, elapsed_time);
      continue;
    }
    elapsed_time += SSSPp(graph, i, output_path);
    tool.infoprint(i, row, graph.interval, graph.stream, elapsed_time);
  }
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  // Output elapsed time
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

void CPU::runAPSPSG(Graph& graph, std::string& output_path) {
  Tool tool;
  float elapsed_time = 0.0;
  int proEntry = 0;
  float time = 0.0f;
  auto row = graph.rows;
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
#pragma omp parallel for
  for (int i = 0; i < row; i++) {
    if (graph.csrB.row_ptr[i] == graph.csrB.row_ptr[i + 1]) {
      ++proEntry;
      tool.infoprint(proEntry, row, graph.interval, graph.stream, elapsed_time);
      continue;
    }
    if (graph.weighted) {
      time = SSSPs(graph, i, output_path);
    } else {
      time = BFSs(graph, i, output_path);
    }
#pragma omp critical
    {
      elapsed_time += time;
      ++proEntry;
    }
    tool.infoprint(proEntry, row, graph.interval, graph.stream, elapsed_time);
  }
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.stream * 1000)
            << std::endl;
}

void CPU::runMSSPSG(Graph& graph, std::string& output_path) {
  float elapsed_time = 0.0f;
  float time = 0.0f;
  int proEntry = 0;
  Tool tool;
  auto row = graph.rows;

  std::vector<float> averageLength(row, 0.0f);

#pragma omp parallel for
  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % row;
    if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
      ++proEntry;
      continue;
    }
    if (graph.weighted) {
      time = SSSPs(graph, i, output_path);
    } else {
      time = BFSs(graph, i, output_path, averageLength);
      // time = BFSs(graph, i, output_path);
    }
#pragma omp critical
    {
      elapsed_time += time;
      ++proEntry;
    }
  }
  float length = tool.averageShortestPath(averageLength.data(), row);
  printf("%-21s%3.5d\n", "Nodes:", row);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time / (graph.stream * 1000));
  printf("%-21s%5.5lf\n", "Average shortest paths Length:", length);
}

void CPU::runMSSPTG(Graph& graph, std::string& output_path) {
  float elapsed_time = 0.0f;
  int proEntry = 0;
  auto row = graph.rows;

  Tool tool;
  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % row;
    if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
      ++proEntry;
      continue;
    }
    if (graph.weighted) {
      elapsed_time += SSSPp(graph, i, output_path);
    } else {
      elapsed_time += BFSp(graph, i, output_path);
    }
    ++proEntry;
  }
  printf("%-21s%3.5d\n", "Nodes:", row);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time / 1000);
}

void CPU::runBFS(Graph& graph, std::string& output_path) {
  int source = graph.source;
  auto row = graph.rows;
  if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }
  float elapsed_time = BFSp(graph, source, output_path);

  printf("%-21s%3.5d\n", "Nodes:", row);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time / 1000);
}

void CPU::runSSSP(Graph& graph, std::string& output_path) {
  int source = graph.source;
  auto row = graph.rows;
  if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }
  float elapsed_time = SSSPp(graph, source, output_path);
  printf("%-21s%3.5d\n", "Nodes:", row);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time / 1000);
}

float CPU::Closeness_Centrality(Graph& graph, int source) {
  int step = 1;
  bool is_converged = false;
  auto row = graph.rows;
  bool* alpha = new bool[row];
  bool* beta = new bool[row];
  int* distance = new int[row];
  float elapsed = 0.0f;
  float closeness_centrality = 0.0f;

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, 0);

#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    alpha[graph.csrB.col[i]] = true;
    distance[graph.csrB.col[i]] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;

    if (!(step % 2))
      if (graph.weighted)
        is_converged = SOVMP(graph, alpha, beta, distance, step);
      else
        is_converged = SOVMP(graph, beta, alpha, distance, step);
    if (is_converged) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  distance[source] = 0;
  Tool tool;
  closeness_centrality =
      (row - 1) / (tool.averageShortestPath(distance, row) * row);

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;

  printf("%-21s%3.5ld\n", "Source:", source);
  printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);

  printf("%-21s%3.5d\n", "Node:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed / (graph.thread * 1000));

  return closeness_centrality;
}

float CPU::Closeness_Centrality_Weighted(Graph& graph, int source) {
  int step = 1;
  bool is_converged = false;
  auto row = graph.rows;
  bool* alpha = new bool[row];
  bool* beta = new bool[row];
  float* distance = new float[row];
  float elapsed = 0.0f;
  float INF = 1.0 * 0xfffffff;
  float closeness_centrality = 0.0f;

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, INF);

#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    distance[graph.csrB.col[i]] = graph.csrB.val[i];
    alpha[graph.csrB.col[i]] = true;
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

  distance[source] = 0;
  Tool tool;
  closeness_centrality =
      (row - 1) / (tool.averageShortestPath(distance, row) * row);

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;

  printf("%-21s%3.5ld\n", "Source:", source);
  printf("%-21s%3.5lf\n", "Closeness Centrality:", closeness_centrality);

  printf("%-21s%3.5d\n", "Node:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed / (graph.thread * 1000));

  return closeness_centrality;
}

float Betweenness_Centrality(Graph& graph,
                             int source,
                             std::string& output_path) {}

float Betweenness_Centrality_Weighted(Graph& graph,
                                      int source,
                                      std::string& output_path) {}

// kernel
float CPU::BFSp(Graph& graph, int source, std::string& output_path) {
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
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    alpha[graph.csrB.col[i]] = true;
    distance[graph.csrB.col[i]] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      is_converged = SOVMP(graph, alpha, beta, distance, step);
    else
      is_converged = SOVMP(graph, beta, alpha, distance, step);
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
    Tool tool;
    tool.outfile(row, distance, source, output_path);
  }

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed;
}

float CPU::BFSs(Graph& graph, int source, std::string& output_path) {
  int step = 1;
  int entry = graph.csrB.row_ptr[source + 1] - graph.csrB.row_ptr[source];
  auto row = graph.rows;
  int* alpha = new int[row];
  int* beta = new int[row];
  int* distance = new int[row];
  float elapsed = 0.0f;

  std::fill_n(distance, row, 0);
  std::fill_n(alpha, row, 0);
  std::fill_n(beta, row, 0);

  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    distance[graph.csrB.col[i]] = 1;
    alpha[i - graph.csrB.row_ptr[source]] = graph.csrB.col[i];
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      entry = SOVM(graph, alpha, beta, distance, step, entry);
    else
      entry = SOVM(graph, beta, alpha, distance, step, entry);
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
    Tool tool;
    tool.outfile(row, distance, source, output_path);
  }

  delete[] alpha;
  alpha = nullptr;
  delete[] beta;
  beta = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed;
}

float CPU::SSSPp(Graph& graph, int source, std::string& output_path) {
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
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    distance[graph.csrB.col[i]] = graph.csrB.val[i];
    alpha[graph.csrB.col[i]] = true;
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
    Tool tool;
    tool.outfile(row, distance, source, output_path);
  }

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed;
}

float CPU::SSSPs(Graph& graph, int source, std::string& output_path) {
  int step = 1;
  int entry = graph.csrB.row_ptr[source + 1] - graph.csrB.row_ptr[source];
  auto row = graph.rows;
  int* alpha = new int[row];
  int* beta = new int[row];
  float* distance = new float[row];
  float elapsed = 0.0f;
  float INF = 1.0 * 0xfffffff;

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, INF);

  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    distance[graph.csrB.col[i]] = graph.csrB.val[i];
    alpha[i - graph.csrB.row_ptr[source]] = graph.csrB.col[i];
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
    Tool tool;
    tool.outfile(row, distance, source, output_path);
  }

  delete[] alpha;
  alpha = nullptr;
  delete[] beta;
  beta = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed;
}

int CPU::SOVM(Graph& graph,
              int*& alpha,
              int*& beta,
              int*& distance,
              int step,
              int entry) {
  int tmpEntry = 0;
  for (int j = 0; j < entry; j++) {
    int start = graph.csrB.row_ptr[alpha[j]];
    int end = graph.csrB.row_ptr[alpha[j] + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (!distance[graph.csrB.col[k]]) {
          distance[graph.csrB.col[k]] = step;
          beta[tmpEntry] = graph.csrB.col[k];
          ++tmpEntry;
        }
      }
    }
  }
  return tmpEntry;
}

int CPU::GOVM(Graph& graph,
              int*& alpha,
              int*& beta,
              float*& distance,
              int entry) {
  int tmpEntry = 0;
  for (int j = 0; j < entry; j++) {
    int start = graph.csrB.row_ptr[alpha[j]];
    int end = graph.csrB.row_ptr[alpha[j] + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        int index = graph.csrB.col[k];
        float tmp = distance[j] + graph.csrB.val[k];
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

bool CPU::SOVMP(Graph& graph,
                bool*& alpha,
                bool*& beta,
                int*& distance,
                int step) {
  bool converged = true;
  auto row = graph.rows;
#pragma omp parallel for
  for (int j = 0; j < row; j++) {
    if (alpha[j]) {
      int start = graph.csrB.row_ptr[j];
      int end = graph.csrB.row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (!distance[graph.csrB.col[k]]) {
            distance[graph.csrB.col[k]] = step;
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

bool CPU::GOVMP(Graph& graph, bool*& alpha, bool*& beta, float*& distance) {
  bool converged = true;
  auto row = graph.rows;
#pragma omp parallel for
  for (int j = 0; j < row; j++) {
    if (alpha[j]) {
      int start = graph.csrB.row_ptr[j];
      int end = graph.csrB.row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          int index = graph.csrB.col[k];
          float tmp = distance[j] + graph.csrB.val[k];
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

}  // namespace DAWN