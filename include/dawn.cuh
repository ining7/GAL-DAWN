#include "dawn.hxx"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace DAWN {
class GPU {
public:
  void runApspGpu(Graph& graph, std::string& output_path);

  void runSsspGpu(Graph& graph, std::string& output_path);

  void runMsspGpu(Graph& graph, std::string& output_path);

  float ssspGpu(DAWN::Graph&               graph,
                int                        source,
                cudaStream_t               streams,
                thrust::device_vector<int> d_row_ptr,
                thrust::device_vector<int> d_col,
                std::string&               output_path);

  float ssspGpuW(DAWN::Graph&                 graph,
                 int                          source,
                 cudaStream_t                 streams,
                 thrust::device_vector<int>   d_row_ptr,
                 thrust::device_vector<int>   d_col,
                 thrust::device_vector<float> d_val,
                 std::string&                 output_path);
  void  Test(DAWN::Graph& graph, std::string& output_path);
};
}  // namespace DAWN

__global__ void SOVM(const int* row_ptr,
                     const int* col,
                     bool*      alpha,
                     bool*      beta,
                     int*       result,
                     int        rows,
                     int        step,
                     bool*      ptr);

__global__ void GOVM(const int*   row_ptr,
                     const int*   col,
                     const float* val,
                     bool*        alpha,
                     bool*        beta,
                     float*       result,
                     int          rows,
                     int          source,
                     bool*        ptr);

__global__ void GOVM(const int*   row_ptr,
                     const int*   col,
                     const float* val,
                     bool*        alpha,
                     bool*        beta,
                     float*       result,
                     int          rows,
                     int          source,
                     bool*        ptr)
{
  ptr[0] = false;
  int j  = blockIdx.x * blockDim.x + threadIdx.x;
  if ((j < rows) && (alpha[j])) {
    int start = row_ptr[j];
    int end   = row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        int   index = col[k];
        float tmp   = result[j] + val[k];
        if (result[index] > tmp) {
          atomicExch(&result[index], tmp);
          beta[index] = true;
          if ((!ptr[0]) && (index != source))
            ptr[0] = true;
        }
      }
    }
  }
}
__global__ void SOVM(const int* row_ptr,
                     const int* col,
                     bool*      alpha,
                     bool*      beta,
                     int*       result,
                     int        rows,
                     int        step,
                     bool*      ptr)
{
  ptr[0] = false;
  int j  = blockIdx.x * blockDim.x + threadIdx.x;
  if ((j < rows) && (alpha[j])) {
    int start = row_ptr[j];
    int end   = row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        int index = col[k];
        if (!result[index]) {
          result[index] = step;
          beta[index]   = true;
          if (!ptr[0])
            ptr[0] = true;
        }
      }
    }
  }
}

float DAWN::GPU::ssspGpuW(DAWN::Graph&                 graph,
                          int                          source,
                          cudaStream_t                 streams,
                          thrust::device_vector<int>   d_row_ptr,
                          thrust::device_vector<int>   d_col,
                          thrust::device_vector<float> d_val,
                          std::string&                 output_path)
{
  int   step = 1;
  float INF  = 1.0 * 0xfffffff;

  thrust::host_vector<bool>  h_alpha(graph.rows, 0);
  thrust::host_vector<bool>  h_beta(graph.rows, 0);
  thrust::host_vector<float> h_result(graph.rows, INF);
  thrust::host_vector<bool>  h_ptr(1, false);

  float elapsed_time = 0.0f;
  omp_set_dynamic(1);
#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    h_alpha[graph.csrB.col[i]]  = true;
    h_result[graph.csrB.col[i]] = graph.csrB.val[i];
  }
  thrust::device_vector<bool>  d_alpha(graph.rows, 0);
  thrust::device_vector<bool>  d_beta(graph.rows, 0);
  thrust::device_vector<float> d_result(graph.rows, INF);
  thrust::device_vector<bool>  d_ptr(1, false);

  thrust::copy(h_alpha.begin(), h_alpha.end(), d_alpha.begin());
  thrust::copy(h_result.begin(), h_result.end(), d_result.begin());

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
  while (step < graph.rows) {
    step++;
    if (!(step % 2)) {
      GOVM<<<num_blocks, block_size, 0, streams>>>(
        d_row_ptr.data().get(), d_col.data().get(), d_val.data().get(),
        d_alpha.data().get(), d_beta.data().get(), d_result.data().get(),
        graph.rows, source, d_ptr.data().get());
      thrust::fill_n(d_alpha.begin(), graph.rows, false);
    } else {
      GOVM<<<num_blocks, block_size, 0, streams>>>(
        d_row_ptr.data().get(), d_col.data().get(), d_val.data().get(),
        d_beta.data().get(), d_alpha.data().get(), d_result.data().get(),
        graph.rows, source, d_ptr.data().get());
      thrust::fill_n(d_beta.begin(), graph.rows, false);
    }
    // thrust::copy_n(d_beta.begin(), graph.rows, d_alpha.begin());
    // thrust::fill_n(d_beta.begin(), graph.rows, false);
    // thrust::copy_n(d_ptr.begin(), 1, h_ptr.begin());
    if (!d_ptr[0]) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  elapsed_time += elapsed.count();

  // printf("Step is [%d]\n", step);

  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    thrust::copy(d_result.begin(), d_result.end(), h_result.begin());
    printf("Start prinft\n");
    Tool tool;
    tool.outfile(graph.rows, h_result, source, output_path);
  }
  cudaFreeAsync(&d_ptr, streams);
  return elapsed_time;
}

float DAWN::GPU::ssspGpu(DAWN::Graph&               graph,
                         int                        source,
                         cudaStream_t               streams,
                         thrust::device_vector<int> d_row_ptr,
                         thrust::device_vector<int> d_col,
                         std::string&               output_path)
{
  int step = 1;

  thrust::host_vector<bool> h_alpha(graph.rows, false);
  thrust::host_vector<bool> h_beta(graph.rows, false);
  thrust::host_vector<int>  h_result(graph.rows, 0);
  thrust::host_vector<bool> h_ptr(1, false);

  float elapsed_time = 0.0f;
  omp_set_dynamic(1);
#pragma omp parallel for
  for (int i = graph.csrB.row_ptr[source]; i < graph.csrB.row_ptr[source + 1];
       i++) {
    h_alpha[graph.csrB.col[i]]  = true;
    h_result[graph.csrB.col[i]] = 1;
  }
  thrust::device_vector<bool> d_alpha(graph.rows, false);
  thrust::device_vector<bool> d_beta(graph.rows, false);
  thrust::device_vector<int>  d_result(graph.rows, 0);
  thrust::device_vector<bool> d_ptr(1, false);

  thrust::copy(h_alpha.begin(), h_alpha.end(), d_alpha.begin());
  thrust::copy(h_alpha.begin(), h_alpha.end(), d_beta.begin());
  thrust::copy(h_result.begin(), h_result.end(), d_result.begin());

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
  while (step < graph.rows) {
    step++;
    if (!(step % 2)) {
      SOVM<<<num_blocks, block_size, 0, streams>>>(
        d_row_ptr.data().get(), d_col.data().get(), d_alpha.data().get(),
        d_beta.data().get(), d_result.data().get(), graph.rows, step,
        d_ptr.data().get());
      thrust::fill_n(d_alpha.begin(), graph.rows, false);
    } else {
      SOVM<<<num_blocks, block_size, 0, streams>>>(
        d_row_ptr.data().get(), d_col.data().get(), d_beta.data().get(),
        d_alpha.data().get(), d_result.data().get(), graph.rows, step,
        d_ptr.data().get());
      thrust::fill_n(d_beta.begin(), graph.rows, false);
    }
    // thrust::copy_n(d_beta.begin(), graph.rows, d_alpha.begin());

    // thrust::copy_n(d_ptr.begin(), 1, h_ptr.begin());
    if (!d_ptr[0]) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  elapsed_time += elapsed.count();

  // printf("Step is [%d]\n", step);

  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    thrust::copy(d_result.begin(), d_result.end(), h_result.begin());
    printf("Start prinft\n");
    Tool tool;
    tool.outfile(graph.rows, h_result, source, output_path);
  }
  cudaFreeAsync(&d_ptr, streams);
  return elapsed_time;
}

void DAWN::GPU::runApspGpu(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  thrust::device_vector<int>   d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int>   d_col(graph.nnz, 0);
  thrust::device_vector<float> d_val(graph.nnz, 0);
  thrust::copy_n(graph.csrB.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csrB.col, graph.nnz, d_col.begin());
  thrust::copy_n(graph.csrB.val, graph.nnz, d_val.begin());
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
    if (graph.weighted) {
      elapsed_time += ssspGpuW(graph, source, streams[cuda_stream], d_row_ptr,
                               d_col, d_val, output_path);
    } else {
      elapsed_time += ssspGpu(graph, source, streams[cuda_stream], d_row_ptr,
                              d_col, output_path);
    }
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

void DAWN::GPU::runMsspGpu(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  thrust::device_vector<int>   d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int>   d_col(graph.nnz, 0);
  thrust::device_vector<float> d_val(graph.nnz, 0);
  thrust::copy_n(graph.csrB.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csrB.col, graph.nnz, d_col.begin());
  thrust::copy_n(graph.csrB.val, graph.nnz, d_val.begin());

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
    if (graph.weighted) {
      elapsed_time += ssspGpuW(graph, source, streams[cuda_stream], d_row_ptr,
                               d_col, d_val, output_path);
    } else {
      elapsed_time += ssspGpu(graph, source, streams[cuda_stream], d_row_ptr,
                              d_col, output_path);
    }
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

void DAWN::GPU::runSsspGpu(DAWN::Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.csrB.row_ptr[source] == graph.csrB.row_ptr[source + 1]) {
    printf("Source [%d] is isolated node\n", source);
    exit(0);
  }

  thrust::device_vector<int>   d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int>   d_col(graph.nnz, 0);
  thrust::device_vector<float> d_val(graph.nnz, 0);
  thrust::copy_n(graph.csrB.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csrB.col, graph.nnz, d_col.begin());
  thrust::copy_n(graph.csrB.val, graph.nnz, d_val.begin());

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  float elapsed_time = 0.0f;
  if (graph.weighted) {
    elapsed_time +=
      ssspGpuW(graph, source, stream, d_row_ptr, d_col, d_val, output_path);
  } else {
    elapsed_time +=
      ssspGpu(graph, source, stream, d_row_ptr, d_col, output_path);
  }

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;

  cudaStreamDestroy(stream);
}
