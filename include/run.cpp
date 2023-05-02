#include "dawn.hxx"
namespace DAWN {
void CPU::runApspFGCsr(Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  Tool  tool;
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;

  for (int i = 0; i < graph.rows; i++) {
    if (graph.csrB.row_ptr[i] == graph.csrB.row_ptr[i + 1]) {
      tool.infoprint(i, graph.rows, graph.interval, graph.thread, elapsed_time);
      continue;
    }
    elapsed_time += ssspPCsr(graph, i, output_path);
    tool.infoprint(i, graph.rows, graph.interval, graph.thread, elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

void CPU::runApspCGCsr(Graph& graph, std::string& output_path)
{
  Tool  tool;
  float elapsed_time = 0.0;
  int   proEntry     = 0;
  tool.outfile(graph.nnz, graph.csrA.col, -1, output_path);
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
#pragma omp parallel for
  for (int i = 0; i < graph.rows; i++) {
    if (graph.csrB.row_ptr[i] == graph.csrB.row_ptr[i + 1]) {
      ++proEntry;
      tool.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                     elapsed_time);
      continue;
    }
    float time_tmp = ssspSCsr(graph, i, output_path);
#pragma omp critical
    {
      elapsed_time += time_tmp;
      ++proEntry;
    }
    tool.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                   elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;
}

void CPU::runSsspCpuCsr(Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  float elapsed_time = ssspPCsr(graph, source, output_path);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

float CPU::ssspPCsr(Graph& graph, int source, std::string& output_path)
{
  int   dim   = 1;
  int   entry = graph.csrB.row_ptr[source + 1] - graph.csrB.row_ptr[source];
  int   entry_last = entry;
  bool* output     = new bool[graph.rows];
  bool* input      = new bool[graph.rows];
  int*  result     = new int[graph.rows];
  // std::fill_n(input, graph.rows, false);
  // std::fill_n(output, graph.rows, false);
  // std::fill_n(result, graph.rows, 0);
#pragma omp parallel for
  for (int j = 0; j < graph.rows; j++) {
    input[j]  = false;
    output[j] = false;
    result[j] = 0;
  }
#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    input[graph.csrB.col[i]]  = true;
    output[graph.csrB.col[i]] = true;
    result[graph.csrB.col[i]] = 1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (dim < graph.dim) {
    dim++;
#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if (graph.csrA.row_ptr[j] == graph.csrA.row_ptr[j + 1])
        continue;
      for (int k = graph.csrA.row_ptr[j]; k < graph.csrA.row_ptr[j + 1]; k++) {
        if (input[graph.csrA.col[k]]) {
          output[j] = true;
          break;
        }
      }
    }
#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if ((result[j] == 0) && (output[j]) && (j != source)) {
        result[j] = dim;
        ++entry;
      }
      input[j]  = output[j];
      output[j] = false;
    }
    if ((entry > entry_last) && (entry < (graph.rows - 1))) {
      entry_last = entry;
      if (entry_last >= (graph.rows - 1))
        break;
    } else {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  graph.entry += entry_last;

  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;

  return elapsed.count();
}

float CPU::ssspSCsr(Graph& graph, int source, std::string& output_path)
{
  int   dim   = 1;
  int   entry = graph.csrB.row_ptr[source + 1] - graph.csrB.row_ptr[source];
  int   entry_last = entry;
  int   entry_max  = graph.rows - 1;
  bool* output     = new bool[graph.rows];
  bool* input      = new bool[graph.rows];
  int*  result     = new int[graph.rows];
  float elapsed    = 0.0f;
  std::fill_n(input, graph.rows, false);
  std::fill_n(output, graph.rows, false);
  std::fill_n(result, graph.rows, 0);
  // for (int j = 0; j < graph.rows; j++) {
  //   input[j]  = false;
  //   output[j] = false;
  //   result[j] = 0;
  // }
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    input[graph.csrB.col[i]]  = true;
    output[graph.csrB.col[i]] = true;
    result[graph.csrB.col[i]] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (dim < graph.dim) {
    dim++;

    for (int j = 0; j < graph.rows; j++) {
      if (graph.csrA.row_ptr[j] == graph.csrA.row_ptr[j + 1])
        continue;
      for (int k = graph.csrA.row_ptr[j]; k < graph.csrA.row_ptr[j + 1]; k++) {
        if (input[graph.csrA.col[k]]) {
          output[j] = true;
          break;
        }
      }
    }
    for (int j = 0; j < graph.rows; j++) {
      if ((result[j] == 0) && (output[j] == true) && (j != source)) {
        result[j] = dim;
        entry++;
      }
      input[j]  = output[j];
      output[j] = false;
    }
    if ((entry > entry_last) && (entry < entry_max)) {
      entry_last = entry;
      if (entry_last >= entry_max)
        break;
    } else {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  graph.entry += entry_last;

  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    printf("Start prinft\n");
    Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;

  return elapsed;
}

void CPU::runApspFGCsm(Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  Tool  tool;
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;

  for (int i = 0; i < graph.rows; i++) {
    if (graph.csmB.row[i] == 0) {
      tool.infoprint(i, graph.rows, graph.interval, graph.thread, elapsed_time);
      continue;
    }
    elapsed_time += ssspPCsm(graph, i, output_path);
    tool.infoprint(i, graph.rows, graph.interval, graph.thread, elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

void CPU::runApspCGCsm(Graph& graph, std::string& output_path)
{
  Tool  tool;
  float elapsed_time = 0.0;
  int   proEntry     = 0;
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
#pragma omp parallel for
  for (int i = 0; i < graph.rows; i++) {
    if (graph.csmB.row[i] == 0) {
      ++proEntry;
      tool.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                     elapsed_time);
      continue;
    }
    float time_tmp = ssspSCsm(graph, i, output_path);
#pragma omp critical
    {
      elapsed_time += time_tmp;
      ++proEntry;
    }
    tool.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                   elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;
}

void CPU::runSsspCpuCsm(Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.csmB.row[source] == 0) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  float elapsed_time = ssspPCsm(graph, source, output_path);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

float CPU::ssspPCsm(Graph& graph, int source, std::string& output_path)
{
  int   dim        = 1;
  int   entry      = graph.csmB.row[source];
  int   entry_last = entry;
  bool* output     = new bool[graph.rows];
  bool* input      = new bool[graph.rows];
  int*  result     = new int[graph.rows];
  // std::fill_n(input, graph.rows, false);
  // std::fill_n(output, graph.rows, false);
  // std::fill_n(result, graph.rows, 0);
#pragma omp parallel for
  for (int j = 0; j < graph.rows; j++) {
    input[j]  = false;
    output[j] = false;
    result[j] = 0;
  }
#pragma omp parallel for
  for (int i = 0; i < graph.csmB.row[source]; i++) {
    input[graph.csmB.col[source][i]]  = true;
    output[graph.csmB.col[source][i]] = true;
    result[graph.csmB.col[source][i]] = 1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (dim < graph.dim) {
    dim++;
#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if (graph.csmB.row[j] == 0)
        continue;
      for (int k = 0; k < graph.csmA.row[j]; k++) {
        if (input[graph.csmA.col[j][k]]) {
          output[j] = true;
          break;
        }
      }
    }
#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if ((result[j] == 0) && (output[j]) && (j != source)) {
        result[j] = dim;
        ++entry;
      }
      input[j]  = output[j];
      output[j] = false;
    }
    if ((entry > entry_last) && (entry < (graph.rows - 1))) {
      entry_last = entry;
      if (entry_last >= (graph.rows - 1))
        break;
    } else {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  graph.entry += entry_last;

  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;

  return elapsed.count();
}

float CPU::ssspSCsm(Graph& graph, int source, std::string& output_path)
{
  int   dim        = 1;
  int   entry      = graph.csmB.row[source];
  int   entry_last = entry;
  bool* output     = new bool[graph.rows];
  bool* input      = new bool[graph.rows];
  int*  result     = new int[graph.rows];
  std::fill_n(input, graph.rows, false);
  std::fill_n(output, graph.rows, false);
  std::fill_n(result, graph.rows, 0);

  for (int i = 0; i < graph.csmB.row[source]; i++) {
    input[graph.csmB.col[source][i]]  = true;
    output[graph.csmB.col[source][i]] = true;
    result[graph.csmB.col[source][i]] = 1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (dim < graph.dim) {
    dim++;
    for (int j = 0; j < graph.rows; j++) {
      if (graph.csmB.row[j] == 0)
        continue;
      for (int k = 0; k < graph.csmA.row[j]; k++) {
        if (input[graph.csmA.col[j][k]]) {
          output[j] = true;
          break;
        }
      }
    }
    for (int j = 0; j < graph.rows; j++) {
      if ((result[j] == 0) && (output[j]) && (j != source)) {
        result[j] = dim;
        entry++;
      }
      input[j]  = output[j];
      output[j] = false;
    }
    if ((entry > entry_last) && (entry < (graph.rows - 1))) {
      entry_last = entry;
      if (entry_last >= (graph.rows - 1))
        break;
    } else {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  graph.entry += entry_last;

  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;

  return elapsed.count();
}

}  // namespace DAWN