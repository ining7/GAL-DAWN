#include "dawn.hxx"
namespace DAWN {
void CPU::runApspTG(Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  Tool  tool;
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;

  for (int i = 0; i < graph.rows; i++) {
    if (graph.csrB.row_ptr[i] == graph.csrB.row_ptr[i + 1]) {
      tool.infoprint(i, graph.rows, graph.interval, graph.stream, elapsed_time);
      continue;
    }
    if (graph.weighted) {
      elapsed_time += ssspPW(graph, i, output_path);
    } else {
      elapsed_time += ssspP(graph, i, output_path);
    }
    tool.infoprint(i, graph.rows, graph.interval, graph.stream, elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

void CPU::runApspSG(Graph& graph, std::string& output_path)
{
  Tool  tool;
  float elapsed_time = 0.0;
  int   proEntry     = 0;
  float time         = 0.0f;
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
#pragma omp parallel for num_threads(graph.stream)
  for (int i = 0; i < graph.rows; i++) {
    if (graph.csrB.row_ptr[i] == graph.csrB.row_ptr[i + 1]) {
      ++proEntry;
      tool.infoprint(proEntry, graph.rows, graph.interval, graph.stream,
                     elapsed_time);
      continue;
    }
    if (graph.weighted) {
      time = ssspSW(graph, i, output_path);
    } else {
      time = ssspS(graph, i, output_path);
    }
#pragma omp critical
    {
      elapsed_time += time;
      ++proEntry;
    }
    tool.infoprint(proEntry, graph.rows, graph.interval, graph.stream,
                   elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.stream * 1000)
            << std::endl;
}

void CPU::runMsspPCpu(Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0f;
  float time         = 0.0f;
  int   proEntry     = 0;
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  Tool tool;

#pragma omp parallel for
  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % graph.rows;
    if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
      ++proEntry;
      printf("Source [%d] is isolated node\n", source);
      tool.infoprint(proEntry, graph.msource.size(), graph.interval,
                     graph.stream, elapsed_time);
      continue;
    }
    if (graph.weighted) {
      time = ssspSW(graph, i, output_path);
    } else {
      time = ssspS(graph, i, output_path);
    }
#pragma omp critical
    {
      elapsed_time += time;
      ++proEntry;
    }
    tool.infoprint(proEntry, graph.msource.size(), graph.interval, graph.stream,
                   elapsed_time);
  }

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.stream * 1000)
            << std::endl;
}

void CPU::runMsspSCpu(Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0f;
  int   proEntry     = 0;
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  Tool tool;

  delete[] graph.csrA.col;
  graph.csrA.col = NULL;
  delete[] graph.csrA.row_ptr;
  graph.csrA.row_ptr = NULL;
  delete[] graph.csrA.val;
  graph.csrA.val = NULL;
  delete[] graph.csrB.val;
  graph.csrB.val = NULL;

  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % graph.rows;
    if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
      ++proEntry;
      printf("Source [%d] is isolated node\n", source);
      tool.infoprint(proEntry, graph.msource.size(), graph.interval,
                     graph.stream, elapsed_time);
      continue;
    }
    if (graph.weighted) {
      elapsed_time += ssspPW(graph, source, output_path);
    } else {
      elapsed_time += ssspP(graph, source, output_path);
    }
    ++proEntry;

    tool.infoprint(proEntry, graph.msource.size(), graph.interval, graph.stream,
                   elapsed_time);
  }

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.stream * 1000)
            << std::endl;
}

void CPU::runSsspCpu(Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }
  float elapsed_time = 0.0f;
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  if (graph.weighted) {
    std::cout << "weighted" << std::endl;
    elapsed_time += ssspPW(graph, source, output_path);
  } else {
    elapsed_time += ssspP(graph, source, output_path);
  }

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

float CPU::ssspP(Graph& graph, int source, std::string& output_path)
{
  int   dim    = 1;
  int   ptr    = false;
  bool* output = new bool[graph.rows];
  bool* input  = new bool[graph.rows];
  int*  result = new int[graph.rows];

  float elapsed = 0.0f;
  std::fill_n(input, graph.rows, false);
  std::fill_n(output, graph.rows, false);
  std::fill_n(result, graph.rows, 0);

#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    input[graph.csrB.col[i]]  = true;
    output[graph.csrB.col[i]] = true;
    result[graph.csrB.col[i]] = 1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (dim < graph.rows) {
    dim++;
    SOVMP(graph, input, ptr, output, result, dim);
    if (!ptr) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  // graph.entry += entry_last;

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

float CPU::ssspPW(Graph& graph, int source, std::string& output_path)
{
  int    step    = 1;
  float* result  = new float[graph.rows];
  float  elapsed = 0.0f;
  float  INF     = 1.0 * 0xfffffff;
  bool   ptr;
  std::fill_n(result, graph.rows, INF);
#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    result[graph.csrB.col[i]] = graph.csrB.val[i];
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < graph.rows) {
    step++;
    ptr = false;
#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if (result[j]) {
        for (int k = graph.csrB.row_ptr[j]; k < graph.csrB.row_ptr[j + 1];
             k++) {
          int index = graph.csrB.col[k];
          if (result[index] > result[j] + graph.csrB.val[k]) {
            result[index] = result[j] + graph.csrB.val[k];
            if (!ptr)
              ptr = true;
          }
        }
      }
    }
    if (!ptr)
      break;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

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

float CPU::ssspS(Graph& graph, int source, std::string& output_path)
{
  int  dim      = 1;
  int  entry    = 0;
  int  alphaPtr = graph.csrB.row_ptr[source + 1] - graph.csrB.row_ptr[source];
  int* alpha    = new int[graph.rows];
  int* delta    = new int[graph.rows];
  int* result   = new int[graph.rows];

  float elapsed = 0.0f;
  std::fill_n(result, graph.rows, 0);
  std::fill_n(alpha, graph.rows, 0);
  std::fill_n(delta, graph.rows, 0);

  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    result[graph.csrB.col[i]]             = 1;
    alpha[i - graph.csrB.row_ptr[source]] = graph.csrB.col[i];
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (dim < graph.rows) {
    dim++;
    SOVMS(graph, alpha, alphaPtr, delta, result, dim);
    entry += alphaPtr;
    if (!alphaPtr) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  graph.entry += entry;

  delete[] alpha;
  alpha = nullptr;
  delete[] delta;
  delta = nullptr;
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

float CPU::ssspSW(Graph& graph, int source, std::string& output_path)
{
  int    step    = 1;
  float* result  = new float[graph.rows];
  float  elapsed = 0.0f;
  float  INF     = 1.0 * 0xfffffff;
  bool   ptr;
  std::fill_n(result, graph.rows, INF);

  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    result[graph.csrB.col[i]] = graph.csrB.val[i];
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < graph.rows) {
    step++;
    ptr = false;
    for (int j = 0; j < graph.rows; j++) {
      if (result[j]) {
        for (int k = graph.csrB.row_ptr[j]; k < graph.csrB.row_ptr[j + 1];
             k++) {
          int index = graph.csrB.col[k];
          if (result[index] > result[j] + graph.csrB.val[k]) {
            result[index] = result[j] + graph.csrB.val[k];
            if (!ptr)
              ptr = true;
          }
        }
      }
    }
    if (!ptr)
      break;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

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

void CPU::BOVM(Graph& graph,
               bool*& input,
               bool*& output,
               int*&  result,
               int    dim,
               bool&  ptr)
{
  std::memmove(input, output, graph.rows * sizeof(bool));
  for (int j = 0; j < graph.rows; j++) {
    if (result[j])
      continue;
    int start = graph.csrA.row_ptr[j];
    int end   = graph.csrA.row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (input[graph.csrA.col[k]]) {
          output[j] = true;
          result[j] = dim;
          if (!ptr)
            ptr = true;
          break;
        }
      }
    }
  }
}

void CPU::SOVMS(Graph& graph,
                int*&  alpha,
                int&   ptr,
                int*&  delta,
                int*&  result,
                int    dim)
{
  int tmpPtr = 0;
  for (int j = 0; j < ptr; j++) {
    int start = graph.csrB.row_ptr[alpha[j]];
    int end   = graph.csrB.row_ptr[alpha[j] + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (!result[graph.csrB.col[k]]) {
          delta[tmpPtr] = graph.csrB.col[k];
          ++tmpPtr;
          result[graph.csrB.col[k]] = dim;
        }
      }
    }
  }
  std::memcpy(alpha, delta, tmpPtr * sizeof(int));
  ptr = tmpPtr;
}

void CPU::SOVMP(Graph& graph,
                bool*& alpha,
                int&   ptr,
                bool*& delta,
                int*&  result,
                int    dim)
{
  ptr = false;
#pragma omp parallel for
  for (int j = 0; j < graph.rows; j++) {
    if (alpha[j]) {
      int start = graph.csrB.row_ptr[j];
      int end   = graph.csrB.row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (!result[graph.csrB.col[k]]) {
            result[graph.csrB.col[k]] = dim;
            if (!ptr)
              ptr = true;
          }
        }
      }
    }
  }
#pragma omp parallel for
  for (int j = 0; j < graph.rows; j++) {
    if (result[j] && (!delta[j])) {
      alpha[j] = true;
      delta[j] = true;
    } else {
      alpha[j] = false;
    }
  }
}

void CPU::runApspTGCsm(Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  Tool  tool;
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  for (int i = 0; i < graph.rows; i++) {
    if (graph.csmB.row[i] == 0) {
      tool.infoprint(i, graph.rows, graph.interval, graph.stream, elapsed_time);
      continue;
    }
    elapsed_time += ssspPCsm(graph, i, output_path);
    tool.infoprint(i, graph.rows, graph.interval, graph.stream, elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

void CPU::runApspSGCsm(Graph& graph, std::string& output_path)
{
  Tool  tool;
  float elapsed_time = 0.0;
  int   proEntry     = 0;
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
#pragma omp parallel for
  for (int i = 0; i < graph.rows; i++) {
    if (graph.csmB.row[i] == 0) {
      ++proEntry;
      tool.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                     elapsed_time);
      continue;
    }
    float time = ssspSCsm(graph, i, output_path);
#pragma omp critical
    {
      elapsed_time += time;
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