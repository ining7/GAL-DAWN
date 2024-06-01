/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-04-21
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/bc.hxx>

struct FloatArray {
  float* data;
  int size;

  FloatArray(int n) : size(n) { data = new float[n](); }

  ~FloatArray() { delete[] data; }

  // 拷贝构造函数，用于OpenMP归约
  FloatArray(const FloatArray& other) : size(other.size) {
    data = new float[size]();
    std::copy(other.data, other.data + size, data);
  }

  // 归约操作的合并函数
  void merge(const FloatArray& other) {
    for (int i = 0; i < size; ++i) {
      data[i] += other.data[i];
    }
  }
};
#pragma omp declare reduction(merge:FloatArray : omp_out.merge(omp_in)) \
    initializer(omp_priv = omp_orig)

float DAWN::BC_CPU::Betweenness_Centrality(DAWN::Graph::Graph_t& graph,
                                           std::string& output_path) {
  float elapsed_time = 0.0;
  auto row = graph.rows;
  FloatArray bc_values(row);
#pragma omp parallel reduction(+ : elapsed_time) reduction(merge : bc_values)
  {
    float local_elapsed_time = 0.0f;
    FloatArray local_bc_values(row);
    for (int i = 0; i < row; i++) {
      if (graph.csr.row_ptr[i] == graph.csr.row_ptr[i + 1]) {
        // DAWN::Tool::infoprint(i, row, graph.interval, graph.stream,
        //                       elapsed_time);
        continue;
      }
      local_elapsed_time = DAWN::BC_CPU::SOVM(graph.csr.row_ptr, graph.csr.col,
                                              row, i, local_bc_values.data);
      elapsed_time += local_elapsed_time;
      // DAWN::Tool::infoprint(i, row, graph.interval, graph.stream,
      // elapsed_time);
    }
  }

  DAWN::Tool::outfile(row, bc_values.data, output_path);
  for (int i = 0; i < row; i++) {
    std::cout << i << " " << std::fixed << std::setprecision(6)
              << bc_values.data[i] << std::endl;
  }

  return (elapsed_time / 1000);
}

float DAWN::BC_CPU::test(DAWN::Graph::Graph_t& graph,
                         std::string& output_path) {
  std::cout << "test" << std::endl;
  float elapsed_time = 0.0;
  auto row = graph.rows;
  float* bc_values = new float[row]();
  float* bc_temp = new float[row]();
  // int i = 2;
  for (int i = 0; i < row; i++) {
    std::fill_n(bc_temp, row, 0.0f);
    elapsed_time +=
        DAWN::BC_CPU::SOVM(graph.csr.row_ptr, graph.csr.col, row, i, bc_temp);
    for (int i = 0; i < row; i++) {
      bc_values[i] += bc_temp[i];
    }
  }

  DAWN::Tool::outfile(row, bc_values, output_path);
  delete[] bc_temp;
  bc_temp = nullptr;
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
      is_converged =
          DAWN::BC_CPU::kernel(row_ptr, col, row, alpha, beta, gamma, amount,
                               path, path_length, bc_temp, step, is_converged);
    else
      is_converged =
          DAWN::BC_CPU::kernel(row_ptr, col, row, beta, alpha, gamma, amount,
                               path, path_length, bc_temp, step, is_converged);
    if (is_converged) {
      break;
    }
  }
  DAWN::BC_CPU::accelerate(path, path_length, bc_temp);
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
                          bool is_converged) {
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
    path[reach].push(through);
    bc_temp[through] += (1.0f / amount[reach]);
    // printf("bc_temp[%d](%f) += (1 / amount[%d])(%f);\n", through,
    //        bc_temp[through], reach, 1.0f / amount[reach]);
    beta[reach] = true;
    if (!gamma[reach]) {
      gamma[reach] = true;
      path_length.push_back(reach);
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

void DAWN::BC_CPU::accelerate(std::vector<std::queue<int>>& path,
                              std::deque<int>& path_length,
                              float*& bc_temp) {
  while (!path_length.empty()) {
    int reach = path_length.back();
    path_length.pop_back();
    while (!path[reach].empty()) {
      int through = path[reach].front();
      bc_temp[through] += bc_temp[reach];
      // printf("{%d,%d}\n", through, reach);
      path[reach].pop();
    }
  }
}