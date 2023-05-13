#include "dawn.hxx"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

__global__ void vecMatOpe(int   rows,
                          int*  d_A_entry,
                          int*  d_A,
                          bool* input,
                          bool* output,
                          int*  result,
                          int   source,
                          int   dim,
                          int*  d_entry);

// create and read
namespace DAWN {
class GPU {
public:
  void runApspGpu(Graph& graph, std::string& output_path);

  void runApspGpuSpmv(Graph& graph, std::string& output_path);

  void runSsspGpu(Graph& graph, std::string& output_path);

  void runSsspGpuSpmv(Graph& graph, std::string& output_path);

  float ssspGpu(Graph&       graph,
                int          source,
                cudaStream_t streams,
                int*         d_A_entry,
                int*         d_A,
                std::string& output_path);

  float ssspGpuSpmv(Graph&                graph,
                    int                   source,
                    std::string&          output_path,
                    cusparseSpMatDescr_t& matA,
                    cusparseDnVecDescr_t& vecX,
                    cusparseDnVecDescr_t& vecY,
                    float*&               d_input,
                    float*&               d_output,
                    void*&                buffers,
                    cudaStream_t          streams,
                    cusparseHandle_t      handle);
};

}  // namespace DAWN

__global__ void vecMatOpe(int   rows,
                          int*  d_A_entry,
                          int*  d_A,
                          bool* input,
                          bool* output,
                          int*  result,
                          int   source,
                          int   dim,
                          int*  d_entry)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < rows && (d_A_entry[j] != d_A_entry[j + 1])) {
    int start = d_A_entry[j];      // 当前行的起始位置
    int end   = d_A_entry[j + 1];  // 当前行的结束位置
    for (int k = start; k < end; k++) {
      if (input[d_A[k]]) {
        output[j] = true;
        if ((result[j] == 0) && (source != j)) {
          result[j]  = dim;
          d_entry[j] = 1;
        }
        break;
      }
    }
  }
  // if (j < rows) {
  //   input[j]  = output[j];
  //   output[j] = false;
  // }
}

void DAWN::GPU::runApspGpu(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  // Copy data to device
  int *d_A_entry, *d_A;
  cudaMalloc((void**)&d_A_entry, sizeof(int) * (graph.rows + 1));
  cudaMalloc((void**)&d_A, sizeof(int) * graph.nnz);

  cout << graph.nnz << endl;
  DAWN dawn;
  dawn.COO2CSR(graph.A, graph.A_entry, graph.csr_row_ptr, graph.csr_col,
               graph.csr_val, graph.rows, graph.nnz);
  std::cerr << "Copy graph" << std::endl;

  cudaMemcpy(d_A_entry, graph.csr_row_ptr, (graph.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, graph.csr_col, graph.nnz * sizeof(int),
             cudaMemcpyHostToDevice);

  // Create streams
  cudaStream_t streams[graph.stream];
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamCreate(&streams[i]);
  }

  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  for (int i = 0; i < graph.rows; i++) {
    int source = i;
    if (graph.B_entry[i] == 0) {
      proEntry++;
      dawn.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                     elapsed_time);
      continue;
    }
    int cuda_stream = source % graph.stream;
    elapsed_time +=
      ssspGpu(graph, source, streams[cuda_stream], d_A_entry, d_A, output_path);
    proEntry++;
    dawn.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
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
  cudaFree(d_A_entry);
  cudaFree(d_A);
}

void DAWN::GPU::runApspGpuSpmv(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  DAWN dawn;
  dawn.COO2CSR(graph.A, graph.A_entry, graph.csr_row_ptr, graph.csr_col,
               graph.csr_val, graph.rows, graph.nnz);
  std::cerr << "Copy graph" << std::endl;

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  int *  dA_csrOffsets, *dA_columns;
  float* dA_values;
  cudaMalloc((void**)&dA_csrOffsets, (graph.rows + 1) * sizeof(int));
  cudaMalloc((void**)&dA_columns, graph.nnz * sizeof(int));
  cudaMalloc((void**)&dA_values, graph.nnz * sizeof(float));

  cudaMemcpy(dA_csrOffsets, graph.csr_row_ptr, (graph.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_columns, graph.csr_col, graph.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_values, graph.csr_val, graph.nnz * sizeof(float),
             cudaMemcpyHostToDevice);

  cusparseSpMatDescr_t matA;
  // Create sparse graph A in CSR format
  cusparseCreateCsr(&matA, graph.rows, graph.cols, graph.nnz, dA_csrOffsets,
                    dA_columns, dA_values, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseDnVecDescr_t* vecX = new cusparseDnVecDescr_t[graph.stream];
  cusparseDnVecDescr_t* vecY = new cusparseDnVecDescr_t[graph.stream];

  float** d_input  = new float*[graph.stream];
  float** d_output = new float*[graph.stream];

  void*        buffers[graph.stream];
  size_t       bufferSizes[graph.stream];
  cudaStream_t streams[graph.stream];

  for (int i = 0; i < graph.stream; ++i) {
    d_input[i]  = new float[graph.cols];
    d_output[i] = new float[graph.cols];
    cusparseCreateDnVec(&vecX[i], graph.cols, d_input[i], CUDA_R_32F);
    cusparseCreateDnVec(&vecY[i], graph.cols, d_output[i], CUDA_R_32F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            matA, vecX[i], &beta, vecY[i], CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizes[i]);
    cudaMalloc(&buffers[i], bufferSizes[i]);
    cudaStreamCreate(&streams[i]);
  }

  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  for (int i = 0; i < graph.rows; i++) {
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    int source = i;
    if (graph.B_entry[i] == 0) {
      proEntry++;
      dawn.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                     elapsed_time);
      continue;
    }
    int cuda_stream = source % graph.stream;
    cout << cuda_stream << endl;
    elapsed_time += ssspGpuSpmv(
      graph, source, output_path, matA, vecX[cuda_stream], vecY[cuda_stream],
      d_input[cuda_stream], d_output[cuda_stream], buffers[cuda_stream],
      streams[cuda_stream], handle);
    proEntry++;
    dawn.infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                   elapsed_time);
    cusparseDestroy(handle);
  }
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;

  // Synchronize streams and Free memory on device
  for (int i = 0; i < graph.stream; i++) {
    cusparseDestroyDnVec(vecX[i]);
    cusparseDestroyDnVec(vecY[i]);
    cudaFree(buffers[i]);
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    delete[] d_input[i];
    delete[] d_output[i];
  }

  delete[] vecX;
  delete[] vecY;
  delete[] d_input;
  delete[] d_output;

  cudaFree(dA_csrOffsets);
  cudaFree(dA_columns);
  cudaFree(dA_values);
  cusparseDestroySpMat(matA);
  cusparseDestroy(handle);
}

void DAWN::GPU::runSsspGpu(DAWN::Graph& graph, std::string& output_path)
{
  int source = graph.source;
  if (graph.B_entry[source] == 0) {
    cout << "Source is isolated node, please check" << endl;
    exit(0);
  }

  // Copy data to device
  int *d_A_entry, *d_A;
  cudaMalloc((void**)&d_A_entry, sizeof(int) * (graph.rows + 1));
  cudaMalloc((void**)&d_A, sizeof(int) * graph.nnz);

  cout << graph.nnz << endl;
  DAWN dawn;
  dawn.COO2CSR(graph.A, graph.A_entry, graph.csr_row_ptr, graph.csr_col,
               graph.csr_val, graph.rows, graph.nnz);

  for (int i = 0; i < graph.rows; i++) {
    for (int j = 0; j < graph.A_entry[i]; j++) {
      // cout << " graph.A[i][j]: " << graph.A[i][j] << endl;
      if (graph.A[i][j] > graph.rows - 1)
        printf("graph.A[%d,%d]=%d \n", i, j, graph.A[i][j]);
    }
    // cout << endl;
  }
  std::cerr << "Copy graph" << std::endl;

  cudaMemcpy(d_A_entry, graph.csr_row_ptr, (graph.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, graph.csr_col, graph.nnz * sizeof(int),
             cudaMemcpyHostToDevice);

  for (int i = 0; i < 10; i++)
    cout << " graph.csr_col[i]: " << graph.csr_col[i] << endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  float elapsed_time =
    ssspGpu(graph, source, stream, d_A_entry, d_A, output_path);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (graph.thread * 1000)
            << std::endl;

  cudaStreamDestroy(stream);
  // Free memory on device
  cudaFree(d_A_entry);
  cudaFree(d_A);
}

void DAWN::GPU::runSsspGpuSpmv(DAWN::Graph& graph, std::string& output_path)
{
  float elapsed_time = 0.0;

  DAWN::Tool tool;
  tool.coo2Csr(graph.A, graph.A_entry, graph.csr_row_ptr, graph.csr_col,
               graph.csr_val, graph.rows, graph.nnz);
  std::cerr << "Copy graph" << std::endl;

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  int *  dA_csrOffsets, *dA_columns;
  float* dA_values;
  cudaMalloc((void**)&dA_csrOffsets, (graph.rows + 1) * sizeof(int));
  cudaMalloc((void**)&dA_columns, graph.nnz * sizeof(int));
  cudaMalloc((void**)&dA_values, graph.nnz * sizeof(float));

  cudaMemcpy(dA_csrOffsets, graph.csr_row_ptr, (graph.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_columns, graph.csr_col, graph.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_values, graph.csr_val, graph.nnz * sizeof(float),
             cudaMemcpyHostToDevice);

  cusparseSpMatDescr_t matA;
  // Create sparse graph A in CSR format
  cusparseCreateCsr(&matA, graph.rows, graph.cols, graph.nnz, dA_csrOffsets,
                    dA_columns, dA_values, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseDnVecDescr_t vecX, vecY;

  float* d_input  = new float[graph.cols];
  float* d_output = new float[graph.cols];

  void*        buffers;
  size_t       bufferSizes;
  cudaStream_t streams;

  cusparseCreateDnVec(&vecX, graph.cols, d_input, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, graph.cols, d_output, CUDA_R_32F);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecY, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizes);
  cudaMalloc(&buffers, bufferSizes);
  cudaStreamCreate(&streams);

  int source = graph.source;
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  if (graph.B_entry[source] == 0) {
    cout << "Source is isolated node, please check" << endl;
    exit(0);
  }
  elapsed_time += ssspGpuSpmv(graph, source, output_path, matA, vecX, vecY,
                              d_input, d_output, buffers, streams, handle);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << "Elapsed Time :"
            << static_cast<double>(elapsed_time) / (graph.thread * 1000) << " s"
            << std::endl;

  // Synchronize streams
  cudaStreamDestroy(streams);

  // Synchronize streams and Free memory on device

  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cudaFree(buffers);
  cudaStreamSynchronize(streams);
  cudaStreamDestroy(streams);
  delete[] d_input;
  delete[] d_output;

  cudaFree(dA_csrOffsets);
  cudaFree(dA_columns);
  cudaFree(dA_values);
  cusparseDestroySpMat(matA);
  cusparseDestroy(handle);
}

float DAWN::GPU::ssspGpu(DAWN::Graph& graph,
                         int          source,
                         cudaStream_t streams,
                         int*         d_A_entry,
                         int*         d_A,
                         std::string& output_path)
{
  int   dim        = 1;
  int   entry      = 0;
  int   entry_last = graph.B_entry[source];
  bool* output     = new bool[graph.rows];
  bool* input      = new bool[graph.rows];
  int*  result     = new int[graph.rows];
  int*  h_entry    = new int[graph.rows];
  omp_set_dynamic(true);
#pragma omp parallel for
  for (int j = 0; j < graph.rows; j++) {
    input[j]   = false;
    output[j]  = false;
    result[j]  = 0;
    h_entry[j] = 0;
  }
#pragma omp parallel for
  for (int i = 0; i < graph.B_entry[source]; i++) {
    input[graph.B[source][i]]  = true;
    output[graph.B[source][i]] = true;
    result[graph.B[source][i]] = 1;
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
    vecMatOpe<<<num_blocks, block_size, 0, streams>>>(
      graph.rows, d_A_entry, d_A, d_input, d_output, d_result, source, dim,
      d_entry);
    cudaMemcpyAsync(h_entry, d_entry, sizeof(int) * graph.rows,
                    cudaMemcpyDeviceToHost, streams);
    for (int i = 0; i < graph.rows; i++)
      if (h_entry[i] > 0)
        cout << " h_entry[i]: " << h_entry[i] << endl;
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
    outfile(graph.rows, result, source, output_path);
  }

  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  delete[] result;
  result = nullptr;

  cudaFreeAsync(d_sum, streams);
  cudaFreeAsync(d_temp_storage, streams);
  cudaFreeAsync(d_input, streams);
  cudaFreeAsync(d_output, streams);
  cudaFreeAsync(d_result, streams);
  cudaFreeAsync(d_entry, streams);

  return elapsed.count();
}

float DAWN::GPU::ssspGpuSpmv(DAWN::Graph&          graph,
                             int                   source,
                             std::string&          output_path,
                             cusparseSpMatDescr_t& matA,
                             cusparseDnVecDescr_t& vecX,
                             cusparseDnVecDescr_t& vecY,
                             float*&               d_input,
                             float*&               d_output,
                             void*&                buffers,
                             cudaStream_t          streams,
                             cusparseHandle_t      handle)
{
  cudaStreamSynchronize(streams);
  // Copy data to device
  cusparseSetStream(handle, streams);
  int         dim        = 1;
  int         entry_last = graph.B_entry[source];
  float       elapsed    = 0.0f;
  const float alpha      = 1.0f;
  const float beta       = 0.0f;

  float* output = new float[graph.rows];
  int*   result = new int[graph.rows];
  std::memset(output, 0, graph.rows * sizeof(float));
  std::memset(result, 0, graph.rows * sizeof(int));
#pragma omp parallel for
  for (int i = 0; i < graph.B_entry[source]; i++) {
    output[graph.B[source][i]] = 1.0f;
    result[graph.B[source][i]] = 1;
  }
  cout << "828" << endl;
  cudaMemcpyAsync(d_input, output, sizeof(float) * graph.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemcpyAsync(d_output, output, sizeof(float) * graph.rows,
                  cudaMemcpyHostToDevice, streams);

  // timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  while (dim < graph.dim) {
    dim++;

    // for (int j = 0; j < graph.rows; j++) {
    //   if (output[j] > 0.0f)
    //     printf("[%d,%d]: %f\n", source, j,
    //     output[j]);
    // }
    if (dim % 2 == 0) {
      // cout << "if dim: " << dim << endl;
      cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
                   &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffers);

      cudaMemcpyAsync(output, d_output, sizeof(float) * graph.rows,
                      cudaMemcpyDeviceToHost, streams);
    }
    // } else {
    //   // cout << "else dim: " << dim << endl;
    //   cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
    //   vecY,
    //                &beta, vecX, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
    //                buffers);

    //   cudaMemcpyAsync(output, d_input, sizeof(float) * graph.rows,
    //                   cudaMemcpyDeviceToHost, streams);
    // }
    // for (int j = 0; j < graph.rows; j++) {
    //   if (output[j] > 0.0f)
    //     printf("[%d,%d]: %f\n", source, j,
    //     output[j]);
    // }
    // cout << endl;
    int entry = 0;
#pragma omp parallel for
    for (int j = 0; j < graph.rows; j++) {
      if ((result[j] == 0) && (output[j] > 0.0f) && (j != source)) {
        result[j] = dim;
        ++entry;
      }
    }
    if ((entry > 0) && (entry < graph.rows - 1)) {
      entry_last += entry;
      if (entry_last >= graph.rows - 1)
        break;
    } else {
      break;
    }
  }
  cudaEventRecord(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  cout << "890" << endl;
  graph.entry += entry_last;

  delete[] output;
  output = nullptr;
  // 输出结果
  if ((graph.prinft) && (source == graph.source)) {
    outfile(graph.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;

  cudaStreamSynchronize(streams);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed;
}
