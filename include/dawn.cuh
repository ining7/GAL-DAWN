#include "dawn.hxx"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace DAWN {
class GPU {
public:
  void runApspGpuCsr(Graph& graph, std::string& output_path);

  void runSsspGpuCsr(Graph& graph, std::string& output_path);

  void runMsspGpuCsr(Graph& graph, std::string& output_path);

  float ssspGpuCsr(DAWN::Graph&               graph,
                   int                        source,
                   cudaStream_t               streams,
                   thrust::device_vector<int> d_row_ptr,
                   thrust::device_vector<int> d_col,
                   std::string&               output_path);
  float SSSPSOVMP(DAWN::Graph& graph,
                  int          source,
                  cudaStream_t streams,
                  int*         d_row_ptr,
                  int*         d_col,
                  std::string& output_path);
  void  Test(DAWN::Graph& graph, std::string& output_path);
};
}  // namespace DAWN

__device__ __managed__ int* d_row_ptr;
__device__ __managed__ int* d_col;

__global__ void BOVMCsr(bool* input,
                        bool* output,
                        int*  result,
                        int*  rows,
                        int*  source,
                        int*  dim,
                        int*  d_entry);
__global__ void BOVMCsrShare(bool* input,
                             bool* output,
                             int*  result,
                             int*  rows,
                             int*  source,
                             int*  dim,
                             int*  d_entry);
__global__ void SOVMCsr(bool* alpha,
                        bool* delta,
                        bool* beta,
                        int*  result,
                        int*  d_row_ptr,
                        int*  d_col,
                        int   rows,
                        int   dim);

__global__ void BOVMCsr(bool* input,
                        bool* output,
                        int*  result,
                        int*  rows,
                        int*  dim,
                        int*  d_entry)
{
  extern __shared__ int shared[];  // 声明共享内存

  shared[0] = *rows;
  shared[2] = *dim;

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < shared[0]) {
    int start = d_row_ptr[j];
    int end   = d_row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (input[d_col[k]]) {
          output[j] = true;
          if (!result[j]) {
            result[j] = shared[2];
            atomicAdd(d_entry, 1);
          }
          break;
        }
      }
    }
  }
}

__global__ void BOVMCsrShare(bool* input,
                             bool* output,
                             int*  result,
                             int*  rows,
                             int*  dim,
                             int*  d_entry)
{
  extern __shared__ int shared[];  // 声明共享内存
  __syncthreads();
  shared[0] = *rows;
  shared[2] = *dim;

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < shared[0]) {
    int start = shared[blockIdx.x] = d_row_ptr[j];
    int end = shared[blockIdx.x + 1] = d_row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (input[d_col[k]]) {
          output[j] = true;
          if (!result[j]) {
            result[j] = shared[2];
            atomicAdd(d_entry, 1);
          }
          break;
        }
      }
    }
  }
}

__global__ void SOVMCsr(bool* alpha,
                        bool* delta,
                        bool* beta,
                        int*  result,
                        int*  d_row_ptr,
                        int*  d_col,
                        int   rows,
                        int   dim)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  if (j < rows && alpha[j]) {
    int start = d_row_ptr[j];
    int end   = d_row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (!delta[d_col[k]]) {
          beta[d_col[k]]   = true;
          result[d_col[k]] = dim;
        }
      }
    }
  }
}

__global__ void SOVMPKernel(const int* row_ptr,
                            const int* col,
                            bool*      alpha,
                            bool*      delta,
                            bool*      beta,
                            int*       result,
                            int        rows,
                            int        dim)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  if (j < rows) {
    if (alpha[j]) {
      int start = row_ptr[j];
      int end   = row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (!delta[col[k]]) {
            beta[col[k]]   = true;
            result[col[k]] = dim;
          }
        }
      }
    }
  }
}

float DAWN::GPU::ssspGpuCsr(DAWN::Graph&               graph,
                            int                        source,
                            cudaStream_t               streams,
                            thrust::device_vector<int> d_row_ptr,
                            thrust::device_vector<int> d_col,
                            std::string&               output_path)
{
  int dim = 1;

  thrust::host_vector<bool> h_alpha(graph.rows, 0);
  thrust::host_vector<bool> h_beta(graph.rows, 0);
  thrust::host_vector<bool> h_delta(graph.rows, 0);
  thrust::host_vector<int>  h_result(graph.rows, 0);

  float elapsed_time = 0.0f;
  omp_set_dynamic(1);
#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    h_alpha[graph.csrB.col[i]]  = true;
    h_beta[graph.csrB.col[i]]   = true;
    h_delta[graph.csrB.col[i]]  = true;
    h_result[graph.csrB.col[i]] = 1;
  }

  thrust::device_vector<bool> d_alpha(graph.rows, 0);
  thrust::device_vector<bool> d_delta(graph.rows, 0);
  thrust::device_vector<bool> d_beta(graph.rows, 0);
  thrust::device_vector<int>  d_result(graph.rows, 0);

  thrust::copy(h_alpha.begin(), h_alpha.end(), d_alpha.begin());
  thrust::copy(h_beta.begin(), h_beta.end(), d_beta.begin());
  thrust::copy(h_delta.begin(), h_delta.end(), d_delta.begin());
  thrust::copy(h_result.begin(), h_result.end(), d_result.begin());

  // Launch kernel
  int block_size      = graph.block_size;
  int num_blocks      = (graph.cols + block_size - 1) / block_size;
  int shared_mem_size = 0;
  if (graph.share) {
    shared_mem_size = sizeof(int) * (4);
  } else {
    shared_mem_size = sizeof(int) * (8);
  }

  while (dim < graph.dim) {
    dim++;
    auto start = std::chrono::high_resolution_clock::now();
    SOVMCsr<<<num_blocks, block_size, shared_mem_size, streams>>>(
      d_alpha.data().get(), d_delta.data().get(), d_beta.data().get(),
      d_result.data().get(), d_row_ptr.data().get(), d_col.data().get(),
      graph.rows, dim);
    // thrust::copy(d_alpha.begin(), d_alpha.end(), h_alpha.begin());
    thrust::copy(d_beta.begin(), d_beta.end(), h_beta.begin());
    // thrust::copy(d_delta.begin(), d_delta.end(), h_delta.begin());
    bool find = 0;

#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if (h_beta[j] && (!h_delta[j])) {
        h_alpha[j] = true;
        h_delta[j] = h_beta[j];
        if (!find)
          find = true;
      } else {
        h_alpha[j] = false;
      }
    }
    if (!find) {
      break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    elapsed_time += elapsed.count();
    thrust::copy(h_alpha.begin(), h_alpha.end(), d_alpha.begin());
    thrust::copy(h_beta.begin(), h_beta.end(), d_beta.begin());
    thrust::copy(h_delta.begin(), h_delta.end(), d_delta.begin());
  }

  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    thrust::copy(d_result.begin(), d_result.end(), h_result.begin());
    printf("Start prinft\n");
    Tool tool;
    tool.outfile(graph.rows, h_result, source, output_path);
  }

  return elapsed_time;
}

float DAWN::GPU::SSSPSOVMP(DAWN::Graph& graph,
                           int          source,
                           cudaStream_t streams,
                           int*         d_row_ptr,
                           int*         d_col,
                           std::string& output_path)
{
  int   dim          = 1;
  bool  alphaPtr     = false;
  bool* delta        = new bool[graph.rows];
  bool* beta         = new bool[graph.rows];
  bool* alpha        = new bool[graph.rows];
  int*  result       = new int[graph.rows];
  float elapsed_time = 0.0f;

  std::fill_n(beta, graph.rows, false);
  std::fill_n(delta, graph.rows, false);
  std::fill_n(result, graph.rows, 0);
  std::fill_n(alpha, graph.rows, false);
  omp_set_dynamic(1);
#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    beta[graph.csrB.col[i]]   = true;
    delta[graph.csrB.col[i]]  = true;
    alpha[graph.csrB.col[i]]  = true;
    result[graph.csrB.col[i]] = 1;
  }
  // Allocate device memory
  bool *d_alpha, *d_delta, *d_beta;
  int*  d_result;
  int   rows = graph.rows;

  // Allocate memory on the device
  cudaMallocAsync((void**)&d_alpha, rows * sizeof(bool), streams);
  cudaMallocAsync((void**)&d_delta, rows * sizeof(bool), streams);
  cudaMallocAsync((void**)&d_beta, rows * sizeof(bool), streams);
  cudaMallocAsync((void**)&d_result, rows * sizeof(int), streams);

  // Copy data from host to device
  cudaMemcpyAsync(d_alpha, alpha, rows * sizeof(bool), cudaMemcpyHostToDevice,
                  streams);
  cudaMemcpyAsync(d_delta, delta, rows * sizeof(bool), cudaMemcpyHostToDevice,
                  streams);
  cudaMemcpyAsync(d_beta, beta, rows * sizeof(bool), cudaMemcpyHostToDevice,
                  streams);
  cudaMemcpyAsync(d_result, result, rows * sizeof(int), cudaMemcpyHostToDevice,
                  streams);

  // Launch kernel
  int block_size = graph.block_size;
  int num_blocks = (graph.cols + block_size - 1) / block_size;
  // int shared_mem_size = 0;
  // if (graph.share) {
  //   shared_mem_size = sizeof(int) * (4);
  // } else {
  //   shared_mem_size = sizeof(int) * (8);
  // }
  auto start = std::chrono::high_resolution_clock::now();
  while (dim < graph.dim) {
    dim++;
    for (int i = 0; i < graph.rows; i++) {
      if (alpha[i] > 0)
        printf("alpha[%d] = %d\n", i, alpha[i]);
    }
    SOVMPKernel<<<num_blocks, block_size, 0, streams>>>(
      d_row_ptr, d_col, d_alpha, d_delta, d_beta, d_result, rows, dim);
    cudaMemcpyAsync(alpha, d_alpha, rows * sizeof(bool), cudaMemcpyDeviceToHost,
                    streams);
    cudaMemcpyAsync(beta, d_beta, rows * sizeof(bool), cudaMemcpyDeviceToHost,
                    streams);
    cudaMemcpyAsync(delta, d_delta, rows * sizeof(bool), cudaMemcpyDeviceToHost,
                    streams);

    for (int i = 0; i < graph.rows; i++) {
      if (alpha[i] > 0)
        printf("alpha[%d] = %d\n", i, alpha[i]);
    }
    // for (int i = 0; i < graph.rows; i++) {
    //   printf("beta[%d] = %d\n", i, beta[i]);
    // }
    // for (int i = 0; i < graph.rows; i++) {
    //   printf("delta[%d] = %d\n", i, delta[i]);
    // }
    alphaPtr = false;
#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if (beta[j] && (!delta[j])) {
        alpha[j] = true;
        delta[j] = beta[j];
        if (!alphaPtr)
          alphaPtr = true;
      } else {
        alpha[j] = false;
      }
    }
    if (!alphaPtr)
      break;
    cudaMemcpyAsync(d_alpha, alpha, rows * sizeof(bool), cudaMemcpyHostToDevice,
                    streams);
    cudaMemcpyAsync(d_delta, delta, rows * sizeof(bool), cudaMemcpyHostToDevice,
                    streams);
    cudaMemcpyAsync(d_beta, beta, rows * sizeof(bool), cudaMemcpyHostToDevice,
                    streams);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  elapsed_time += elapsed.count();

  // Free device memory
  cudaFreeAsync(d_alpha, streams);
  cudaFreeAsync(d_delta, streams);
  cudaFreeAsync(d_beta, streams);
  cudaFreeAsync(d_result, streams);
  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] delta;
  delta = nullptr;
  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    cudaMemcpy(result, d_result, rows * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Start prinft\n");
    Tool tool;
    tool.outfile(graph.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;
  return elapsed_time;
}

void DAWN::GPU::Test(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  cudaMalloc((void**)&d_row_ptr, (graph.rows + 1) * sizeof(int));
  cudaMalloc((void**)&d_col, graph.nnz * sizeof(int));
  cudaMemcpy(d_row_ptr, graph.csrB.row_ptr, (graph.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col, graph.csrB.col, graph.nnz * sizeof(int),
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
    elapsed_time += SSSPSOVMP(graph, source, streams[cuda_stream], d_row_ptr,
                              d_col, output_path);
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
  cudaFree(d_row_ptr);
  cudaFree(d_col);
}

void DAWN::GPU::runApspGpuCsr(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  thrust::device_vector<int> d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int> d_col(graph.nnz, 0);
  thrust::copy_n(graph.csrB.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csrB.col, graph.nnz, d_col.begin());
  // for (int i = 0; i < 30; i++)
  //   printf("d_col[%d] = %d\n", i, h_col[i]);
  // for (int i = 0; i < 30; i++)
  //   printf("d_row_ptr[%d] = %d\n", i, h_row_ptr[i]);

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
    elapsed_time += ssspGpuCsr(graph, source, streams[cuda_stream], d_row_ptr,
                               d_col, output_path);
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
}

void DAWN::GPU::runMsspGpuCsr(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  thrust::device_vector<int> d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int> d_col(graph.nnz, 0);
  thrust::copy_n(graph.csrB.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csrB.col, graph.nnz, d_col.begin());

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
    elapsed_time += ssspGpuCsr(graph, source, streams[cuda_stream], d_row_ptr,
                               d_col, output_path);
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
}

void DAWN::GPU::runSsspGpuCsr(DAWN::Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.csrB.row_ptr[source] == 0) {
    std::cout << "Source is isolated node, please check" << std::endl;
    exit(0);
  }

  thrust::device_vector<int> d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int> d_col(graph.nnz, 0);
  thrust::copy_n(graph.csrB.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csrB.col, graph.nnz, d_col.begin());

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  float elapsed_time =
    ssspGpuCsr(graph, source, stream, d_row_ptr, d_col, output_path);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;

  cudaStreamDestroy(stream);
}
