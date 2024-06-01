/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-04-21
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/bc.hxx>

float DAWN::BC_CPU::Betweenness_Centrality(DAWN::Graph::Graph_t& graph,
                                           std::string& output_path) {
  std::cout << "test" << std::endl;
  float elapsed_time = 0.0;
  auto row = graph.rows;
  float* bc_values = new float[row]();

  // int i = 0;
  // #pragma omp parallel for
  for (int i = 0; i < row; i++) {
    float* bc_temp = new float[row]();
    float time =
        DAWN::BC_CPU::SOVM(graph.csr.row_ptr, graph.csr.col, row, i, bc_temp);
    // #pragma omp critical
    {
      elapsed_time += time;
      for (int j = 0; j < row; j++) {
        bc_values[j] += bc_temp[j];
      }
    }
    delete[] bc_temp;
    bc_temp = nullptr;
  }

  DAWN::Tool::outfile(row, bc_values, output_path);

  delete[] bc_values;
  bc_values = nullptr;

  return (elapsed_time / 1000);
}

float DAWN::BC_CPU::SOVM(int* row_ptr,
                         int* col,
                         int row,
                         int source,
                         float*& bc_temp) {
  int step = 1;
  float elapsed = 0.0f;
  bool is_converged = false;
  bool* alpha = new bool[row]();
  bool* beta = new bool[row]();
  bool* gamma = new bool[row]();
  int* amount = new int[row]();
  std::vector<std::queue<int>> path(row);
  std::deque<int> path_length;

  for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) {
    alpha[col[i]] = true;
    gamma[col[i]] = true;
    // amount[col[i]] = 1;
    path[col[i]].push(source);
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      is_converged = DAWN::BC_CPU::kernel(row_ptr, col, row, alpha, beta, gamma,
                                          amount, path, path_length, bc_temp,
                                          step, is_converged, source);
    else
      is_converged = DAWN::BC_CPU::kernel(row_ptr, col, row, beta, alpha, gamma,
                                          amount, path, path_length, bc_temp,
                                          step, is_converged, source);
    if (is_converged) {
      break;
    }
  }

  // std::vector<std::queue<int>> temp = path;
  // for (int i = 0; i < row; i++) {
  //   if (i != source) {
  //     printf("{start}={%d}\n", i);
  //     while (!temp[i].empty()) {
  //       int through = temp[i].front();
  //       printf("{start, reach}={%d,%d}\n", through, i);
  //       temp[i].pop();
  //     }
  //   }
  // }
  // printf("Source={%d}\n", source);
  // for (int k = 0; k < row; k++) {
  //   if (bc_temp[k]) {
  //     printf("bc_temp[%d] = %f\n", k, bc_temp[k]);
  //   }
  // }
  DAWN::BC_CPU::accumulate(path, path_length, bc_temp, source);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  delete[] alpha;
  alpha = nullptr;
  delete[] beta;
  beta = nullptr;
  delete[] gamma;
  gamma = nullptr;
  delete[] amount;
  amount = nullptr;

  return elapsed;
}

bool DAWN::BC_CPU::kernel(int* row_ptr,
                          int* col,
                          int row,
                          bool*& alpha,
                          bool*& beta,
                          bool*& gamma,
                          int*& amount,
                          std::vector<std::queue<int>>& path,
                          std::deque<int>& path_length,
                          float*& bc_temp,
                          int step,
                          bool is_converged,
                          int source) {
  std::queue<std::pair<int, int>> tmp;
  bool converged = true;
  int j = 0;

  for (int j = 0; j < row; j++) {
    if (alpha[j]) {
      int start = row_ptr[j];
      int end = row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (!gamma[col[k]]) {
            amount[col[k]] += 1;
            tmp.push({j, col[k]});
            converged = false;
          }
        }
      }
      alpha[j] = false;
    }
  }

  while (!tmp.empty()) {
    int through = tmp.front().first;
    int reach = tmp.front().second;
    if (reach != source) {
      path[reach].push(through);
      bc_temp[through] += (1.0f / amount[reach]);
      // printf("bc_temp[%d](%f) += (1 / amount[%d])(%f);\n", through,
      //        bc_temp[through], reach, 1.0f / amount[reach]);
      beta[reach] = true;
      if (!gamma[reach]) {
        gamma[reach] = true;
        path_length.push_back(reach);
      }
    }

    // printf("{%d,%d}\n", through, reach);
    tmp.pop();
  }
  // for (int k = 0; k < row; k++) {
  //   if (bc_temp[k]) {
  //     printf("step[%d]:bc_temp[%d] = %f\n", step, k, bc_temp[k]);
  //   }
  // }
  return converged;
}

void DAWN::BC_CPU::accumulate(std::vector<std::queue<int>>& path,
                              std::deque<int>& path_length,
                              float*& bc_temp,
                              int source) {
  while (!path_length.empty()) {
    int reach = path_length.back();
    path_length.pop_back();
    while (!path[reach].empty()) {
      int through = path[reach].front();
      // if (through != source) {
      bc_temp[through] += bc_temp[reach];
      // printf(
      //     "{through, reach, bc_temp[reach], bc_temp[through] "
      //     "}{%d,%d,%lf,%lf}\n",
      //     through, reach, bc_temp[reach], bc_temp[through]);
      path[reach].pop();
    }
    // }
  }
}