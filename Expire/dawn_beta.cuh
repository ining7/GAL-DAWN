#include "dawn.hxx"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace DAWN {
class GPU {
public:
  void runApspGpuCsr(Graph& graph, std::string& output_path);

  void runSsspGpuCsr(Graph& graph, std::string& output_path);

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

__device__ __managed__ int** d_A;
__global__ void              vecMatOpeCsr(int   rows,
                                          int*  d_A_row_ptr,
                                          int*  d_A_col,
                                          bool* input,
                                          bool* output,
                                          int*  result,
                                          int   source,
                                          int   dim,
                                          int*  d_entry);
__global__ void              vecMatOpemanaged(int   rows,
                                              int*  d_A_row,
                                              bool* input,
                                              bool* output,
                                              int*  result,
                                              int   source,
                                              int   dim,
                                              int*  d_entry);

__global__ void vecMatOpeCsr(int   rows,
                             int*  d_A_row_ptr,
                             int*  d_A_col,
                             bool* input,
                             bool* output,
                             int*  result,
                             int   source,
                             int   dim,
                             int*  d_entry)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < rows && (d_A_row_ptr[j] != d_A_row_ptr[j + 1])) {
    int start = d_A_row_ptr[j];      // 当前行的起始位置
    int end   = d_A_row_ptr[j + 1];  // 当前行的结束位置
    for (int k = start; k < end; k++) {
      if (input[d_A_col[k]]) {
        output[j] = true;
        if ((result[j] == 0) && (source != j)) {
          result[j]  = dim;
          d_entry[j] = 1;
        }
        break;
      }
    }
  }
}

__global__ void vecMatOpemanaged(int   rows,
                                 int*  d_A_row,
                                 bool* input,
                                 bool* output,
                                 int*  result,
                                 int   source,
                                 int   dim,
                                 int*  d_entry)
{
  int j     = blockIdx.x * blockDim.x + threadIdx.x;
  int entry = 0;
  if (j < rows && (d_A_row[j] != 0)) {
    for (int k = 0; k < d_A_row[j]; k++) {
      if (input[d_A[j][k]]) {
        output[j] = true;
        if ((result[j] == 0) && (source != j)) {
          result[j] = dim;
          ++entry;
        }
        break;
      }
    }
  }
  // if (j < rows) {
  //   input[j]  = output[j];
  //   output[j] = false;
  // }
  atomicAdd(d_entry, entry);
}
void DAWN::GPU::runApspGpuCsr(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  // Copy data to device
  int *d_A_row_ptr, *d_A_col;
  cudaMalloc((void**)&d_A_row_ptr, sizeof(int) * (graph.rows + 1));
  cudaMalloc((void**)&d_A_col, sizeof(int) * graph.nnz);
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
    if (graph.csrB.row_ptr[i] == graph.csrB.row_ptr[i + 1]) {
      ++proEntry;
      tool.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                     elapsed_time);
      continue;
    }

    int source      = i;
    int cuda_stream = source % graph.stream;
    // printf("i = %d\n", i);
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

void DAWN::GPU::runSsspGpuCsr(DAWN::Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.csrB.row_ptr[source] == 0) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }

  // Copy data to device
  int *d_A_row_ptr, *d_A_col;
  cudaMalloc((void**)&d_A_row_ptr, sizeof(int) * (graph.rows + 1));
  cudaMalloc((void**)&d_A_col, sizeof(int) * graph.nnz);
  std::cerr << "Copy graph" << std::endl;

  cudaMemcpy(d_A_row_ptr, graph.csrA.row_ptr, (graph.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_col, graph.csrA.col, graph.nnz * sizeof(int),
             cudaMemcpyHostToDevice);

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
  int dim        = 1;
  int entry      = 0;
  int entry_last = graph.csrB.row_ptr[source + 1] - graph.csrB.row_ptr[source];
  bool* output   = new bool[graph.rows];
  bool* input    = new bool[graph.rows];
  int*  result   = new int[graph.rows];
  int*  h_entry  = new int[graph.rows];
  omp_set_dynamic(true);
#pragma omp parallel for
  for (int j = 0; j < graph.rows; j++) {
    input[j]   = false;
    output[j]  = false;
    result[j]  = 0;
    h_entry[j] = 0;
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
  int*  d_entry;

  cudaMallocAsync((void**)&d_input, sizeof(bool) * graph.cols, streams);
  cudaMallocAsync((void**)&d_output, sizeof(bool) * graph.rows, streams);
  cudaMallocAsync((void**)&d_result, sizeof(int) * graph.rows, streams);
  cudaMallocAsync((void**)&d_entry, sizeof(int) * graph.rows, streams);

  cudaMemcpyAsync(d_input, input, sizeof(bool) * graph.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemcpyAsync(d_output, output, sizeof(bool) * graph.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemsetAsync(d_entry, 0, sizeof(int) * graph.rows, streams);
  cudaMemcpyAsync(d_result, result, sizeof(int) * graph.rows,
                  cudaMemcpyHostToDevice, streams);

  // Launch kernel
  int block_size = graph.block_size;
  int num_blocks = (graph.cols + block_size - 1) / block_size;

  void*  d_temp_storage     = NULL;
  size_t temp_storage_bytes = 0;
  int*   d_sum              = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_entry, d_sum,
                         graph.rows, streams);
  cudaMallocAsync(&d_sum, sizeof(int), streams);
  cudaMallocAsync(&d_temp_storage, temp_storage_bytes, streams);

  auto start = std::chrono::high_resolution_clock::now();
  while (dim < graph.dim) {
    dim++;
    // for (int i = 0; i < graph.rows; i++)
    //   if (input[i] > 0)
    //     printf("input[%d] = %d\n", i, input[i]);
    vecMatOpeCsr<<<num_blocks, block_size, 0, streams>>>(
      graph.rows, d_A_row_ptr, d_A_col, d_input, d_output, d_result, source,
      dim, d_entry);

    cudaMemcpyAsync(h_entry, d_entry, sizeof(int) * graph.rows,
                    cudaMemcpyDeviceToHost, streams);
    // for (int i = 0; i < graph.rows; i++)
    //   if (h_entry[i] > 0)
    //     std::cout << " h_entry[i]: " << h_entry[i] << std::endl;
    cudaStreamSynchronize(streams);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_entry, d_sum,
                           graph.rows, streams);
    cudaMemcpyAsync(&entry, d_sum, sizeof(int), cudaMemcpyDeviceToHost,
                    streams);
    // cout << " entry: " << entry << endl;
    cudaMemcpyAsync(d_input, d_output, sizeof(bool) * graph.rows,
                    cudaMemcpyDeviceToDevice, streams);
    cudaMemsetAsync(d_output, 0, sizeof(bool) * graph.rows, streams);
    cudaMemsetAsync(d_entry, 0, sizeof(int) * graph.rows, streams);

    if ((entry > 0) && (entry < (graph.rows - 1))) {
      entry_last += entry;
      if (entry_last >= (graph.rows - 1))
        break;
    } else {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  cudaMemcpyAsync(result, d_result, sizeof(int) * graph.rows,
                  cudaMemcpyDeviceToHost, streams);
  graph.entry += entry_last;

  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    DAWN::Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }
  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  delete[] result;
  result = nullptr;
  delete[] h_entry;
  result = nullptr;

  cudaFreeAsync(d_sum, streams);
  cudaFreeAsync(d_temp_storage, streams);
  cudaFreeAsync(d_input, streams);
  cudaFreeAsync(d_output, streams);
  cudaFreeAsync(d_result, streams);
  cudaFreeAsync(d_entry, streams);

  return elapsed.count();
}