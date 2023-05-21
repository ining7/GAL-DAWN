#include "dawn.hxx"

namespace DAWN {
class GPU {
public:
  void runApspGpuCsr(Graph& graph, std::string& output_path);

  void runSsspGpuCsr(Graph& graph, std::string& output_path);

  void runMsspGpuCsr(Graph& graph, std::string& output_path);

  float ssspGpuCsr(Graph&       graph,
                   int          source,
                   cudaStream_t streams,
                   int*         d_A_row_ptr,
                   int*         d_A_col,
                   std::string& output_path);

  void runApspGpuCsm(Graph& graph, std::string& output_path);

  void runSsspGpuCsm(Graph& graph, std::string& output_path);

  float ssspGpuCsm(Graph&       graph,
                   int          source,
                   cudaStream_t streams,
                   int*         d_A_row_ptr,
                   int*         d_A_col,
                   std::string& output_path);
};
}  // namespace DAWN

__device__ __managed__ int* d_A_row_ptr;
__device__ __managed__ int* d_A_col;

__global__ void vecMatOpe(bool* input,
                          bool* output,
                          int*  result,
                          int*  rows,
                          int*  source,
                          int*  dim,
                          int*  d_entry);
__global__ void vecMatOpeCsr(bool* input,
                             bool* output,
                             int*  result,
                             int*  rows,
                             int*  source,
                             int*  dim,
                             int*  d_entry);
__global__ void vecMatOpeCsrShare(bool* input,
                                  bool* output,
                                  int*  result,
                                  int*  rows,
                                  int*  source,
                                  int*  dim,
                                  int*  d_entry);

__global__ void vecMatOpe(bool* input,
                          bool* output,
                          int*  result,
                          int*  rows,
                          int*  source,
                          int*  dim,
                          int*  d_entry)
{
  int d_row    = *rows;
  int d_source = *source;
  int d_dim    = *dim;
  int j        = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d_row) {
    int start = d_A_row_ptr[j];      // 当前行的起始位置
    int end   = d_A_row_ptr[j + 1];  // 当前行的结束位置
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (input[d_A_col[k]]) {
          output[j] = true;
          if ((result[j] == 0) && (j != d_source)) {
            result[j] = d_dim;
            atomicAdd(d_entry, 1);
          }
          break;
        }
      }
    }
  }
}

__global__ void vecMatOpeCsr(bool* input,
                             bool* output,
                             int*  result,
                             int*  rows,
                             int*  source,
                             int*  dim,
                             int*  d_entry)
{
  extern __shared__ int shared[];  // 声明共享内存

  shared[0] = *rows;
  shared[1] = *source;
  shared[2] = *dim;

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < shared[0]) {
    int start = d_A_row_ptr[j];
    int end   = d_A_row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (input[d_A_col[k]]) {
          output[j] = true;
          if ((result[j] == 0) && (j != shared[1])) {
            result[j] = shared[2];
            atomicAdd(d_entry, 1);
          }
          break;
        }
      }
    }
  }
}

__global__ void vecMatOpeCsrShare(bool* input,
                                  bool* output,
                                  int*  result,
                                  int*  rows,
                                  int*  source,
                                  int*  dim,
                                  int*  d_entry)
{
  extern __shared__ int shared[];  // 声明共享内存
  __syncthreads();
  shared[0] = *rows;
  shared[1] = *source;
  shared[2] = *dim;

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < shared[0]) {
    int start = shared[blockIdx.x] = d_A_row_ptr[j];
    int end = shared[blockIdx.x + 1] = d_A_row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (input[d_A_col[k]]) {
          output[j] = true;
          if ((result[j] == 0) && (shared[1] != j)) {
            result[j] = shared[2];
            atomicAdd(d_entry, 1);
          }
          break;
        }
      }
    }
  }
}

void DAWN::GPU::runApspGpuCsr(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  // Copy data to device
  cudaMallocManaged((void**)&d_A_row_ptr, sizeof(int) * (graph.rows + 1));
  cudaMallocManaged((void**)&d_A_col, sizeof(int) * graph.nnz);
  std::cerr << "Copy graph" << std::endl;

  cudaMemcpy(d_A_row_ptr, graph.csrA.row_ptr, (graph.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_col, graph.csrA.col, graph.nnz * sizeof(int),
             cudaMemcpyHostToDevice);

  // Create streams
  cudaStream_t streams[graph.stream];
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamCreate(&streams[i]);
  }

  Tool tool;
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  for (int i = 0; i < graph.rows; i++) {
    int source = i;
    if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
      ++proEntry;
      // printf("Source [%d] is isolated node\n", source);
      tool.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                     elapsed_time);
      continue;
    }
    int cuda_stream = source % graph.stream;
    elapsed_time += ssspGpuCsr(graph, source, streams[cuda_stream], d_A_row_ptr,
                               d_A_col, output_path);
    ++proEntry;
    tool.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                   elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;

  // Synchronize streams
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  // Free memory on device
  cudaFree(d_A_row_ptr);
  cudaFree(d_A_col);
}

void DAWN::GPU::runMsspGpuCsr(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  cudaMallocManaged((void**)&d_A_row_ptr, sizeof(int) * (graph.rows + 1));
  cudaMallocManaged((void**)&d_A_col, sizeof(int) * graph.nnz);
  std::cerr << "Copy graph" << std::endl;

  cudaMemcpy(d_A_row_ptr, graph.csrA.row_ptr, (graph.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_col, graph.csrA.col, graph.nnz * sizeof(int),
             cudaMemcpyHostToDevice);

  // Create streams
  cudaStream_t streams[graph.stream];
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamCreate(&streams[i]);
  }

  Tool tool;
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;

  for (int i = 0; i < graph.msource.size(); i++) {
    int source = graph.msource[i] % graph.rows;
    if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
      ++proEntry;
      printf("Source [%d] is isolated node\n", source);
      tool.infoprint(proEntry, graph.msource.size(), graph.interval,
                     graph.thread, elapsed_time);
      continue;
    }
    int cuda_stream = source % graph.stream;
    elapsed_time += ssspGpuCsr(graph, source, streams[cuda_stream], d_A_row_ptr,
                               d_A_col, output_path);
    ++proEntry;
    tool.infoprint(proEntry, graph.msource.size(), graph.interval, graph.thread,
                   elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;

  // Synchronize streams
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  // Free memory on device
  cudaFree(d_A_row_ptr);
  cudaFree(d_A_col);
}

void DAWN::GPU::runSsspGpuCsr(DAWN::Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.csrB.row_ptr[source] == 0) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }

  // Copy data to device
  int *d_A_row_ptr, *d_A_col;

  cudaMallocManaged((void**)&d_A_row_ptr, sizeof(int) * (graph.rows + 1));
  cudaMallocManaged((void**)&d_A_col, sizeof(int) * graph.nnz);
  std::cerr << "Copy graph" << std::endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  float elapsed_time =
    ssspGpuCsr(graph, source, stream, d_A_row_ptr, d_A_col, output_path);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;

  cudaStreamDestroy(stream);

  // Free memory on device
  cudaFree(d_A_row_ptr);
  cudaFree(d_A_col);
}

float DAWN::GPU::ssspGpuCsr(DAWN::Graph& graph,
                            int          source,
                            cudaStream_t streams,
                            int*         d_A_row_ptr,
                            int*         d_A_col,
                            std::string& output_path)
{
  int   dim   = 1;
  int   entry = graph.csrB.row_ptr[source + 1] - graph.csrB.row_ptr[source];
  int   entry_last   = entry;
  bool* output       = new bool[graph.rows];
  bool* input        = new bool[graph.rows];
  int*  result       = new int[graph.rows];
  float elapsed_time = 0.0f;
  omp_set_dynamic(true);
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

  bool *d_input, *d_output;
  int*  d_result;
  int * d_entry, *d_source, *d_dim, *d_row;

  cudaMallocAsync((void**)&d_input, sizeof(bool) * graph.cols, streams);
  cudaMallocAsync((void**)&d_output, sizeof(bool) * graph.rows, streams);
  cudaMallocAsync((void**)&d_result, sizeof(int) * graph.rows, streams);
  cudaMallocAsync((void**)&d_entry, sizeof(int), streams);
  cudaMallocAsync((void**)&d_source, sizeof(int), streams);
  cudaMallocAsync((void**)&d_row, sizeof(int), streams);
  cudaMallocAsync((void**)&d_dim, sizeof(int), streams);

  cudaMemcpyAsync(d_input, input, sizeof(bool) * graph.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemcpyAsync(d_output, output, sizeof(bool) * graph.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemcpyAsync(d_result, result, sizeof(int) * graph.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemcpyAsync(d_entry, &entry_last, sizeof(int), cudaMemcpyHostToDevice,
                  streams);
  cudaMemcpyAsync(d_source, &source, sizeof(int), cudaMemcpyHostToDevice,
                  streams);
  cudaMemcpyAsync(d_row, &graph.rows, sizeof(int), cudaMemcpyHostToDevice,
                  streams);

  // Launch kernel
  int block_size      = graph.block_size;
  int num_blocks      = (graph.cols + block_size - 1) / block_size;
  int shared_mem_size = 0;
  if (graph.share) {
    shared_mem_size = sizeof(int) * (4 + num_blocks * 2);
  } else {
    shared_mem_size = sizeof(int) * (4);
  }

  while (dim < graph.dim) {
    dim++;
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice, streams);
    if (graph.share) {
      vecMatOpeCsrShare<<<num_blocks, block_size, shared_mem_size, streams>>>(
        d_input, d_output, d_result, d_row, d_source, d_dim, d_entry);
    } else {
      cudaStreamSynchronize(streams);
      vecMatOpeCsr<<<num_blocks, block_size, shared_mem_size, streams>>>(
        d_input, d_output, d_result, d_row, d_source, d_dim, d_entry);
    }
    cudaMemcpyAsync(&entry, d_entry, sizeof(int), cudaMemcpyDeviceToHost,
                    streams);
    if ((entry > entry_last) && (entry < (graph.rows - 1))) {
      entry_last = entry;
      if (entry_last >= (graph.rows - 1))
        break;
    } else {
      break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    elapsed_time += elapsed.count();

    cudaMemcpyAsync(d_input, d_output, sizeof(bool) * graph.rows,
                    cudaMemcpyDeviceToDevice, streams);
    cudaMemsetAsync(d_output, false, sizeof(bool) * graph.rows, streams);
  }

  graph.entry += entry_last;

  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    cudaMemcpyAsync(result, d_result, sizeof(int) * graph.rows,
                    cudaMemcpyDeviceToHost, streams);
    DAWN::Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }

  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  delete[] result;
  result = nullptr;

  cudaFreeAsync(d_input, streams);
  cudaFreeAsync(d_output, streams);
  cudaFreeAsync(d_result, streams);
  cudaFreeAsync(d_entry, streams);

  return elapsed_time;
}