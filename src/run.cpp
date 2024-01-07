#include <dawn/dawn.hxx>
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
#pragma omp parallel for
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

void CPU::runMsspP(Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0f;
  float time         = 0.0f;
  int   proEntry     = 0;
  // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> MSSP start "
  //              "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  //           << std::endl;
  Tool tool;

  std::vector<float> averageLength(graph.rows, 0.0f);

#pragma omp parallel for
  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % graph.rows;
    if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
      ++proEntry;
      // printf("Source [%d] is isolated node\n", source);
      // tool.infoprint(proEntry, graph.msource.size(), graph.interval,
      //  graph.stream, elapsed_time);
      continue;
    }
    if (graph.weighted) {
      time = ssspSW(graph, i, output_path);
    } else {
      time = ssspS(graph, i, output_path, averageLength);
    }
#pragma omp critical
    {
      elapsed_time += time;
      ++proEntry;
    }
    // tool.infoprint(proEntry, graph.msource.size(), graph.interval,
    // graph.stream,
    //  elapsed_time);
  }

  // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> MSSP end "
  //              "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  //           << std::endl;
  // // Output elapsed time and free remaining resources
  // std::cout << " Elapsed time: " << elapsed_time / (graph.stream * 1000)
  //           << std::endl;
  float length = tool.averageShortestPath(averageLength.data(), graph.rows);
  printf("%-21s%3.5d\n", "Nodes:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time / (graph.stream * 1000));
  printf("%-21s%5.5lf\n", "Average shortest paths Length:", length);
}

void CPU::runMsspS(Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0f;
  int   proEntry     = 0;
  // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> MSSP start "
  //              "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  //           << std::endl;
  Tool tool;

  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % graph.rows;
    if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
      ++proEntry;
      // printf("Source [%d] is isolated node\n", source);
      // tool.infoprint(proEntry, graph.msource.size(), graph.interval,
      //                graph.stream, elapsed_time);
      continue;
    }
    if (graph.weighted) {
      elapsed_time += ssspPW(graph, source, output_path);
    } else {
      elapsed_time += ssspP(graph, source, output_path);
    }
    ++proEntry;

    // tool.infoprint(proEntry, graph.msource.size(), graph.interval,
    // graph.stream,elapsed_time);
  }

  // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> MSSP end "
  //              "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  //           << std::endl;
  // Output elapsed time and free remaining resources
  printf("%-21s%3.5d\n", "Nodes:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time / 1000);
}

void CPU::runSssp(Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }
  float elapsed_time = 0.0f;
  if (graph.weighted) {
    std::cout << "weighted" << std::endl;
    elapsed_time += ssspPW(graph, source, output_path);
  } else {
    elapsed_time += ssspP(graph, source, output_path);
  }
  // Output elapsed time
  // std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
  printf("%-21s%3.5d\n", "Nodes:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time / 1000);
}

float CPU::ssspP(Graph& graph, int source, std::string& output_path)
{
  int   step   = 1;
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
  while (step < graph.rows) {
    step++;
    SOVMP(graph, input, ptr, output, result, step);
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
  // Output
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
  bool*  alpha   = new bool[graph.rows];
  bool*  beta    = new bool[graph.rows];
  float  elapsed = 0.0f;
  float  INF     = 1.0 * 0xfffffff;
  bool   ptr;
  std::fill_n(result, graph.rows, INF);
  std::fill_n(alpha, graph.rows, false);
  std::fill_n(beta, graph.rows, false);

#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    result[graph.csrB.col[i]] = graph.csrB.val[i];
    alpha[graph.csrB.col[i]]  = true;
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < graph.rows) {
    step++;
    ptr = false;
#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if (alpha[j]) {
        for (int k = graph.csrB.row_ptr[j]; k < graph.csrB.row_ptr[j + 1];
             k++) {
          int   index = graph.csrB.col[k];
          float tmp   = result[j] + graph.csrB.val[k];
          if (result[index] > tmp) {
            result[index] = std::min(result[index], tmp);
            beta[index]   = true;
            if ((!ptr) && (index != source))
              ptr = true;
          }
        }
      }
    }
    std::copy_n(beta, graph.rows, alpha);
    std::fill_n(beta, graph.rows, false);
    if (!ptr)
      break;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  // printf("Step is [%d]\n", step);

  // Output
  if ((graph.prinft) && (source == graph.source)) {
    printf("Start prinft\n");
    Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }
  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] result;
  result = nullptr;
  return elapsed;
}

float CPU::ssspS(Graph& graph, int source, std::string& output_path)
{
  int  step     = 1;
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
  while (step < graph.rows) {
    step++;
    SOVMS(graph, alpha, alphaPtr, delta, result, step);
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
  // Output
  if ((graph.prinft) && (source == graph.source)) {
    printf("Start prinft\n");
    Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }

  delete[] result;
  result = nullptr;

  return elapsed;
}

float CPU::ssspS(Graph&              graph,
                 int                 source,
                 std::string&        output_path,
                 std::vector<float>& averageLenth)
{
  int  step     = 1;
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
  while (step < graph.rows) {
    step++;
    SOVMS(graph, alpha, alphaPtr, delta, result, step);
    entry += alphaPtr;
    if (!alphaPtr) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  graph.entry += entry;

  Tool tool;
  averageLenth[source] = tool.averageShortestPath(result, graph.rows);

  delete[] alpha;
  alpha = nullptr;
  delete[] delta;
  delta = nullptr;
  // Output
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
  int step = 1;

  float* result  = new float[graph.rows];
  bool*  alpha   = new bool[graph.rows];
  bool*  beta    = new bool[graph.rows];
  float  elapsed = 0.0f;
  float  INF     = 1.0 * 0xfffffff;
  bool   ptr;
  std::fill_n(result, graph.rows, INF);
  std::fill_n(alpha, graph.rows, false);
  std::fill_n(beta, graph.rows, false);

  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    result[graph.csrB.col[i]] = graph.csrB.val[i];
    alpha[graph.csrB.col[i]]  = true;
  }

  while (step < graph.rows) {
    step++;
    ptr        = false;
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < graph.rows; j++) {
      if (alpha[j]) {
        for (int k = graph.csrB.row_ptr[j]; k < graph.csrB.row_ptr[j + 1];
             k++) {
          int   index = graph.csrB.col[k];
          float tmp   = result[j] + graph.csrB.val[k];
          if (result[index] > tmp) {
            result[index] = tmp;
            beta[index]   = true;
            if ((!ptr) && (index != source))
              ptr = true;
          }
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
    elapsed += elapsed_tmp.count();
    std::memmove(alpha, beta, graph.rows);
    std::fill_n(beta, graph.rows, false);
    if (!ptr)
      break;
  }

  // printf("Step is [%d]\n", step);

  // Output
  if ((graph.prinft) && (source == graph.source)) {
    printf("Start prinft\n");
    Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }
  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] result;
  result = nullptr;
  return elapsed;
}

void CPU::BOVM(Graph& graph,
               bool*& input,
               bool*& output,
               int*&  result,
               bool&  ptr,
               int    step)
{
  std::memmove(input, output, graph.rows * sizeof(bool));
  for (int j = 0; j < graph.rows; j++) {
    if (!result[j]) {
      int start = graph.csrA.row_ptr[j];
      int end   = graph.csrA.row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (input[graph.csrA.col[k]]) {
            output[j] = true;
            result[j] = step;
            if (!ptr)
              ptr = true;
            break;
          }
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
                int    step)
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
          result[graph.csrB.col[k]] = step;
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
                int    step)
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
            result[graph.csrB.col[k]] = step;
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

}  // namespace DAWN