#include "access.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
using namespace std;

__global__ void vecMatOpe(int   rows,
                          int*  d_A_entry,
                          int*  d_A,
                          bool* input,
                          bool* output,
                          int*  result,
                          int   source,
                          int   dim,
                          int*  d_entry);

class DAWN {
public:
  struct Matrix
  {
    int      rows;
    int      cols;
    uint64_t nnz;
    Csr      csr;
    int*     coo_row;
    int*     coo_col;
    float*   coo_val;
    int*     csr_row_ptr;  // CSR行指针
    int*     csr_col;      // CSR列索引
    float*   csr_val;      // CSR值
    int      dim;
    uint64_t entry;
    int      thread;
    int      interval;
    int      stream;
    int      block_size;
    bool     prinft;  // 是否打印结果
    int      source;  // 打印的节点
  };

  // create and read

  void createGraph(string& input_path, Matrix& matrix);

  void createGraphconvert(string& input_path,
                          Matrix& matrix,
                          string& col_output_path,
                          string& row_output_path);

  void readGraph(string&                 input_path,
                 Matrix&                 matrix,
                 vector<pair<int, int>>& cooMatCol);

  // run
  void runApspV3(Matrix& matrix, string& output_path);

  void runApspV4(Matrix& matrix, string& output_path);

  void runApspGpu(DAWN::Matrix& matrix, std::string& output_path);

  void runApspGpuSpmv(DAWN::Matrix& matrix, std::string& output_path);

  // sssp_kernel
  float ssspP(DAWN::Matrix& matrix, int source, std::string& output_path);

  float sssp(DAWN::Matrix& matrix, int source, std::string& output_path);

  float ssspGpu(DAWN::Matrix& matrix,
                int           source,
                cudaStream_t  streams,
                int*          d_A_entry,
                int*          d_A,
                std::string&  output_path);

  float ssspGpuSpmv(DAWN::Matrix&         matrix,
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

  // big
  void readCRC(Matrix& matrix, string& input_path);

  void readRCC(Matrix& matrix, string& input_path);

  void readGraphBig(string& input_path,
                    string& col_input_path,
                    string& row_input_path,
                    Matrix& matrix);

  // convert

  void COO2RCCconvert(Matrix&                 matrix,
                      vector<pair<int, int>>& cooMatRow,
                      string&                 row_output_path);

  void COO2CRCconvert(Matrix&                 matrix,
                      vector<pair<int, int>>& cooMatCol,
                      string&                 col_output_path);

  void COO2CRC(Matrix& matrix, vector<pair<int, int>>& cooMatCol);

  void COO2RCC(Matrix& matrix, vector<pair<int, int>>& cooMatRow);

  void COO2CSR(int**&  matrix,
               int*&   entry,
               int*&   csr_row_ptr,
               int*&   csr_col,
               float*& csr_val,
               int     n,
               int     nnz);

  // info and output
  void
  infoprint(int entry, int total, int interval, int thread, float elapsed_time);

  void outfile(int n, int* result, int source, std::string& output_path);

  // SSSP run
  void runSsspCpu(DAWN::Matrix& matrix, std::string& output_path);

  void runSsspGpu(DAWN::Matrix& matrix, std::string& output_path);

  void runSsspGpuSpmv(DAWN::Matrix& matrix, std::string& output_path);
};

void coo2csr(int     n,
             int     nnz,
             float*& val,
             int*&   row,
             int*&   col,
             float*& csrval,
             int*&   csrrowptr,
             int*&   csrcolidx)
{
  csrval    = new float[nnz];
  csrrowptr = new int[n + 1];
  csrcolidx = new int[nnz];
  // 统计每一行的非零元素数目
  int* row_count = new int[n];

  for (int i = 0; i < nnz; i++) {
    row_count[row[i]]++;
  }

  // 计算每一行第一个非零元素在csrval和csrcolidx中的位置索引
  csrrowptr[0] = 0;
  for (int i = 1; i <= n; i++) {
    csrrowptr[i] = csrrowptr[i - 1] + row_count[i - 1];
  }

  // 将每个非零元素填充到csrval和csrcolidx中
  for (int i = 0; i < nnz; i++) {
    // 获取当前非零元素的行和列
    int cur_row = row[i];
    int cur_col = col[i];

    // 找到该非零元素在CSR格式中的位置
    int dest = csrrowptr[cur_row];

    // 将非零元素存储到CSR格式的数组中
    csrval[dest]    = val[i];
    csrcolidx[dest] = cur_col;

    // 将csrrowptr数组当前行号所在位置后移一位
    csrrowptr[cur_row]++;
  }

  // 修正csrrowptr
  for (int i = n; i >= 1; i--) {
    csrrowptr[i] = csrrowptr[i - 1];
  }
  csrrowptr[0] = 0;
}

void DAWN::COO2CRC(DAWN::Matrix& matrix, std::vector<pair<int, int>>& cooMatCol)
{
  int         col_a = 0;
  int         k     = 0;
  vector<int> tmp;
  tmp.clear();
  while (k < cooMatCol.size()) {
    if (matrix.A_entry[col_a] != 0) {
      if (cooMatCol[k].second == col_a) {
        tmp.push_back(cooMatCol[k].first);
        k++;
      } else {
#pragma omp parallel for
        for (int j = 0; j < matrix.A_entry[col_a]; j++) {
          matrix.A[col_a][j] = tmp[j];
        }
        tmp.clear();
        col_a++;
      }
    } else {
      col_a++;
    }
  }
#pragma omp parallel for
  for (int j = 0; j < matrix.A_entry[col_a]; j++) {
    matrix.A[col_a][j] = tmp[j];
  }
}

void DAWN::COO2RCC(DAWN::Matrix& matrix, std::vector<pair<int, int>>& cooMatRow)
{
  int         row_b = 0;
  int         k     = 0;
  vector<int> tmp;
  tmp.clear();
  while (k < cooMatRow.size()) {
    if (matrix.B_entry[row_b] != 0) {
      if (cooMatRow[k].first == row_b) {
        tmp.push_back(cooMatRow[k].second);
        k++;
      } else {
#pragma omp parallel for
        for (int j = 0; j < matrix.B_entry[row_b]; j++) {
          matrix.B[row_b][j] = tmp[j];
        }
        tmp.clear();
        row_b++;
      }
    } else {
      row_b++;
    }
  }
#pragma omp parallel for
  for (int j = 0; j < matrix.B_entry[row_b]; j++) {
    matrix.B[row_b][j] = 0;
    matrix.B[row_b][j] = tmp[j];
  }
}

void DAWN::COO2CSR(int**&  matrix,
                   int*&   entry,
                   int*&   csr_row_ptr,
                   int*&   csr_col,
                   float*& csr_val,
                   int     n,
                   int     nnz)
{
  // 初始化CSR的行指针
  csr_row_ptr = new int[n + 1];
  csr_col     = new int[nnz];
  csr_val     = new float[nnz];
  int tmp     = 0;
  for (int i = 0; i < n; i++) {
    if (entry[i] == 0) {
      entry[i] = entry[i - 1];
      continue;
    }
    for (int j = 0; j < entry[i]; j++) {
      csr_col[tmp] = matrix[i][j];
      csr_val[tmp] = 1.0;
      tmp++;
    }
    entry[i]           = tmp;
    csr_row_ptr[i + 1] = tmp;
  }
  csr_row_ptr[0] = 0;
  // for (int i = 0; i < n + 1; i++) {
  //   cout << "matrix.csr_row_ptr[i]: " << csr_row_ptr[i] << endl;
  // }
  // for (int i = 0; i < nnz; i++) {
  //   cout << "matrix.csr_col[i]: " << csr_col[i] << endl;
  // }
  // for (int i = 0; i < nnz; i++) {
  //   cout << "matrix.csr_val[i]: " <<csr_val[i] << endl;
  // }
}

void DAWN::createGraph(std::string& input_path, DAWN::Matrix& matrix)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }

  std::string line;
  int         rows, cols, nnz, dim;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;

    std::stringstream ss(line);
    ss >> rows >> cols >> nnz >> dim;
    break;
  }
  cout << rows << " " << cols << " " << nnz << " " << dim << endl;
  file.close();

  matrix.rows = rows;
  matrix.cols = cols;
  matrix.dim  = dim;

  matrix.coo_row = new int[nnz];
  matrix.coo_col = new int[nnz];
  matrix.coo_val = new float[nnz];
  matrix.entry   = 0;
#pragma omp parallel for
  for (int i = 0; i < matrix.rows; i++) {
    matrix.A_entry[i] = 0;
    matrix.B_entry[i] = 0;
  }

  vector<pair<int, int>> cooMatCol;
  cout << "Read Input Graph" << endl;
  readGraph(input_path, matrix, cooMatCol);

  cout << "Create Transport Graph" << endl;
  vector<pair<int, int>> cooMatRow = cooMatCol;
  sort(cooMatRow.begin(), cooMatRow.end());

  cout << "Create Input Matrices" << endl;
#pragma omp parallel for
  for (int i = 0; i < matrix.rows; i++) {
    matrix.A[i] = new int[matrix.A_entry[i]];
    matrix.B[i] = new int[matrix.B_entry[i]];
  }
  COO2CRC(matrix, cooMatCol);
  COO2RCC(matrix, cooMatRow);
  matrix.nnz = 0;
  for (int i = 0; i < matrix.rows; i++) {
    matrix.nnz += matrix.A_entry[i];
  }

  cout << "Initialize Input Matrices" << endl;
}

void DAWN::readGraph(std::string&                 input_path,
                     DAWN::Matrix&                matrix,
                     std::vector<pair<int, int>>& cooMatCol)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  int rows, cols;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols;
    rows--;
    cols--;
    if (rows != cols) {
      cooMatCol.push_back({rows, cols});
      matrix.A_entry[cols]++;
      matrix.B_entry[rows]++;
      matrix.entry++;
    }
  }
  file.close();

  matrix.nnz = matrix.entry;
  cout << "nnz: " << matrix.nnz << endl;
}

void DAWN::runApspV3(DAWN::Matrix& matrix, std::string& output_path)
{
  float elapsed_time = 0.0;

  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;

  for (int i = 0; i < matrix.rows; i++) {
    if (matrix.B_entry[i] == 0) {
      infoprint(i, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
      continue;
    }
    elapsed_time += ssspP(matrix, i, output_path);
    infoprint(i, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

void DAWN::runApspV4(DAWN::Matrix& matrix, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
#pragma omp parallel for
  for (int i = 0; i < matrix.rows; i++) {
    if (matrix.B_entry[i] == 0) {
#pragma omp critical
      {
        ++proEntry;
        infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread,
                  elapsed_time);
      }
      continue;
    }
    float time_tmp = sssp(matrix, i, output_path);
#pragma omp critical
    {
      elapsed_time += time_tmp;
      ++proEntry;
    }
    infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread,
              elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (matrix.thread * 1000)
            << std::endl;
}

void DAWN::runApspGpu(DAWN::Matrix& matrix, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  // Copy data to device
  int *d_A_entry, *d_A;
  cudaMalloc((void**)&d_A_entry, sizeof(int) * (matrix.rows + 1));
  cudaMalloc((void**)&d_A, sizeof(int) * matrix.nnz);

  cout << matrix.nnz << endl;
  DAWN dawn;
  dawn.COO2CSR(matrix.A, matrix.A_entry, matrix.csr_row_ptr, matrix.csr_col,
               matrix.csr_val, matrix.rows, matrix.nnz);
  std::cerr << "Copy Matrix" << std::endl;

  cudaMemcpy(d_A_entry, matrix.csr_row_ptr, (matrix.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, matrix.csr_col, matrix.nnz * sizeof(int),
             cudaMemcpyHostToDevice);

  // Create streams
  cudaStream_t streams[matrix.stream];
  for (int i = 0; i < matrix.stream; i++) {
    cudaStreamCreate(&streams[i]);
  }

  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  for (int i = 0; i < matrix.rows; i++) {
    int source = i;
    if (matrix.B_entry[i] == 0) {
      proEntry++;
      dawn.infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread,
                     elapsed_time);
      continue;
    }
    int cuda_stream = source % matrix.stream;
    elapsed_time += ssspGpu(matrix, source, streams[cuda_stream], d_A_entry,
                            d_A, output_path);
    proEntry++;
    dawn.infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread,
                   elapsed_time);
  }
  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (matrix.thread * 1000)
            << std::endl;

  // Synchronize streams
  for (int i = 0; i < matrix.stream; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  // Free memory on device
  cudaFree(d_A_entry);
  cudaFree(d_A);
}

void DAWN::runApspGpuSpmv(DAWN::Matrix& matrix, std::string& output_path)
{
  float elapsed_time = 0.0;
  int   proEntry     = 0;

  DAWN dawn;
  dawn.COO2CSR(matrix.A, matrix.A_entry, matrix.csr_row_ptr, matrix.csr_col,
               matrix.csr_val, matrix.rows, matrix.nnz);
  std::cerr << "Copy Matrix" << std::endl;

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  int *  dA_csrOffsets, *dA_columns;
  float* dA_values;
  cudaMalloc((void**)&dA_csrOffsets, (matrix.rows + 1) * sizeof(int));
  cudaMalloc((void**)&dA_columns, matrix.nnz * sizeof(int));
  cudaMalloc((void**)&dA_values, matrix.nnz * sizeof(float));

  cudaMemcpy(dA_csrOffsets, matrix.csr_row_ptr, (matrix.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_columns, matrix.csr_col, matrix.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_values, matrix.csr_val, matrix.nnz * sizeof(float),
             cudaMemcpyHostToDevice);

  cusparseSpMatDescr_t matA;
  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, matrix.rows, matrix.cols, matrix.nnz, dA_csrOffsets,
                    dA_columns, dA_values, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseDnVecDescr_t* vecX = new cusparseDnVecDescr_t[matrix.stream];
  cusparseDnVecDescr_t* vecY = new cusparseDnVecDescr_t[matrix.stream];

  float** d_input  = new float*[matrix.stream];
  float** d_output = new float*[matrix.stream];

  void*        buffers[matrix.stream];
  size_t       bufferSizes[matrix.stream];
  cudaStream_t streams[matrix.stream];

  for (int i = 0; i < matrix.stream; ++i) {
    d_input[i]  = new float[matrix.cols];
    d_output[i] = new float[matrix.cols];
    cusparseCreateDnVec(&vecX[i], matrix.cols, d_input[i], CUDA_R_32F);
    cusparseCreateDnVec(&vecY[i], matrix.cols, d_output[i], CUDA_R_32F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            matA, vecX[i], &beta, vecY[i], CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizes[i]);
    cudaMalloc(&buffers[i], bufferSizes[i]);
    cudaStreamCreate(&streams[i]);
  }

  std::cout
    << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  for (int i = 0; i < matrix.rows; i++) {
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    int source = i;
    if (matrix.B_entry[i] == 0) {
      proEntry++;
      dawn.infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread,
                     elapsed_time);
      continue;
    }
    int cuda_stream = source % matrix.stream;
    cout << cuda_stream << endl;
    elapsed_time += ssspGpuSpmv(
      matrix, source, output_path, matA, vecX[cuda_stream], vecY[cuda_stream],
      d_input[cuda_stream], d_output[cuda_stream], buffers[cuda_stream],
      streams[cuda_stream], handle);
    proEntry++;
    dawn.infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread,
                   elapsed_time);
    cusparseDestroy(handle);
  }
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (matrix.thread * 1000)
            << std::endl;

  // Synchronize streams and Free memory on device
  for (int i = 0; i < matrix.stream; i++) {
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

float DAWN::ssspP(DAWN::Matrix& matrix, int source, std::string& output_path)
{
  int   dim        = 1;
  int   entry      = matrix.B_entry[source];
  int   entry_last = entry;
  bool* output     = new bool[matrix.rows];
  bool* input      = new bool[matrix.rows];
  int*  result     = new int[matrix.rows];
  int   entry_max  = matrix.rows - 1;
#pragma omp parallel for
  for (int j = 0; j < matrix.rows; j++) {
    input[j]  = false;
    output[j] = false;
    result[j] = 0;
  }
#pragma omp parallel for
  for (int i = 0; i < matrix.B_entry[source]; i++) {
    input[matrix.B[source][i]]  = true;
    output[matrix.B[source][i]] = true;
    result[matrix.B[source][i]] = 1;
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (dim < matrix.dim) {
    dim++;
#pragma omp parallel for
    for (int j = 0; j < matrix.rows; j++) {
      if (matrix.A_entry[j] == 0)
        continue;
      for (int k = 0; k < matrix.A_entry[j]; k++) {
        if (input[matrix.A[j][k]] == true) {
          output[j] = true;
          break;
        }
      }
    }
#pragma omp parallel for
    for (int j = 0; j < matrix.rows; j++) {
      if ((result[j] == 0) && (output[j] == true) && (j != source)) {
        result[j] = dim;
#pragma omp atomic
        entry++;
      }
      input[j]  = output[j];
      output[j] = false;
    }
    if ((entry > entry_last) && (entry < matrix.rows - 1)) {
      entry_last = entry;
      if (entry_last >= entry_max)
        break;
    } else {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  matrix.entry += entry_last;

  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  // 输出结果
  if ((matrix.prinft) && (source == matrix.source)) {
    outfile(matrix.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;

  return elapsed.count();
}

float DAWN::sssp(DAWN::Matrix& matrix, int source, std::string& output_path)
{
  int   dim        = 1;
  int   entry      = matrix.B_entry[source];
  int   entry_last = entry;
  bool* output     = new bool[matrix.rows];
  bool* input      = new bool[matrix.rows];
  int*  result     = new int[matrix.rows];
  int   entry_max  = matrix.rows - 1;
  for (int j = 0; j < matrix.rows; j++) {
    input[j]  = false;
    output[j] = false;
    result[j] = 0;
  }

  for (int i = 0; i < matrix.B_entry[source]; i++) {
    input[matrix.B[source][i]]  = true;
    output[matrix.B[source][i]] = true;
    result[matrix.B[source][i]] = 1;
  }
  auto start = std::chrono::high_resolution_clock::now();

  while (dim < matrix.dim) {
    dim++;

    for (int j = 0; j < matrix.rows; j++) {
      if (matrix.A_entry[j] == 0)
        continue;
      for (int k = 0; k < matrix.A_entry[j]; k++) {
        if (input[matrix.A[j][k]] == true) {
          output[j] = true;
          break;
        }
      }
    }
    for (int j = 0; j < matrix.rows; j++) {
      if ((result[j] == 0) && (output[j] == true) && (j != source)) {
        result[j] = dim;
        entry++;
      }
      input[j]  = output[j];
      output[j] = false;
    }
    if ((entry > entry_last) && (entry < matrix.rows - 1)) {
      entry_last = entry;
      if (entry_last >= entry_max)
        break;
    } else {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  matrix.entry += entry_last;

  delete[] output;
  output = nullptr;
  delete[] input;
  input = nullptr;
  // 输出结果
  if ((matrix.prinft) && (source == matrix.source)) {
    outfile(matrix.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;

  return elapsed.count();
}

float DAWN::ssspGpu(DAWN::Matrix& matrix,
                    int           source,
                    cudaStream_t  streams,
                    int*          d_A_entry,
                    int*          d_A,
                    std::string&  output_path)
{
  int   dim        = 1;
  int   entry      = 0;
  int   entry_last = matrix.B_entry[source];
  bool* output     = new bool[matrix.rows];
  bool* input      = new bool[matrix.rows];
  int*  result     = new int[matrix.rows];
  int*  h_entry    = new int[matrix.rows];
  omp_set_dynamic(true);
#pragma omp parallel for
  for (int j = 0; j < matrix.rows; j++) {
    input[j]   = false;
    output[j]  = false;
    result[j]  = 0;
    h_entry[j] = 0;
  }
#pragma omp parallel for
  for (int i = 0; i < matrix.B_entry[source]; i++) {
    input[matrix.B[source][i]]  = true;
    output[matrix.B[source][i]] = true;
    result[matrix.B[source][i]] = 1;
  }

  bool *d_input, *d_output;
  int*  d_result;
  int*  d_entry;

  cudaMallocAsync((void**)&d_input, sizeof(bool) * matrix.cols, streams);
  cudaMallocAsync((void**)&d_output, sizeof(bool) * matrix.rows, streams);
  cudaMallocAsync((void**)&d_result, sizeof(int) * matrix.rows, streams);
  cudaMallocAsync((void**)&d_entry, sizeof(int) * matrix.rows, streams);

  cudaMemcpyAsync(d_input, input, sizeof(bool) * matrix.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemcpyAsync(d_output, output, sizeof(bool) * matrix.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemsetAsync(d_entry, 0, sizeof(int) * matrix.rows, streams);
  cudaMemcpyAsync(d_result, result, sizeof(int) * matrix.rows,
                  cudaMemcpyHostToDevice, streams);

  // Launch kernel
  int block_size = matrix.block_size;
  int num_blocks = (matrix.cols + block_size - 1) / block_size;

  void*  d_temp_storage     = NULL;
  size_t temp_storage_bytes = 0;
  int*   d_sum              = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_entry, d_sum,
                         matrix.rows, streams);
  cudaMallocAsync(&d_sum, sizeof(int), streams);
  cudaMallocAsync(&d_temp_storage, temp_storage_bytes, streams);

  auto start = std::chrono::high_resolution_clock::now();
  while (dim < matrix.dim) {
    dim++;
    vecMatOpe<<<num_blocks, block_size, 0, streams>>>(
      matrix.rows, d_A_entry, d_A, d_input, d_output, d_result, source, dim,
      d_entry);
    cudaMemcpyAsync(h_entry, d_entry, sizeof(int) * matrix.rows,
                    cudaMemcpyDeviceToHost, streams);
    for (int i = 0; i < matrix.rows; i++)
      if (h_entry[i] > 0)
        cout << " h_entry[i]: " << h_entry[i] << endl;
    cudaStreamSynchronize(streams);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_entry, d_sum,
                           matrix.rows, streams);
    cudaMemcpyAsync(&entry, d_sum, sizeof(int), cudaMemcpyDeviceToHost,
                    streams);
    // cout << " entry: " << entry << endl;
    cudaMemcpyAsync(d_input, d_output, sizeof(bool) * matrix.rows,
                    cudaMemcpyDeviceToDevice, streams);
    cudaMemsetAsync(d_output, 0, sizeof(bool) * matrix.rows, streams);
    cudaMemsetAsync(d_entry, 0, sizeof(int) * matrix.rows, streams);

    if ((entry > 0) && (entry < (matrix.rows - 1))) {
      entry_last += entry;
      if (entry_last >= (matrix.rows - 1))
        break;
    } else {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  cudaMemcpyAsync(result, d_result, sizeof(int) * matrix.rows,
                  cudaMemcpyDeviceToHost, streams);
  matrix.entry += entry_last;

  // 输出结果
  if ((matrix.prinft) && (source == matrix.source)) {
    outfile(matrix.rows, result, source, output_path);
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

float DAWN::ssspGpuSpmv(DAWN::Matrix&         matrix,
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
  cout << "812" << endl;
  int         dim        = 1;
  int         entry_last = matrix.B_entry[source];
  float       elapsed    = 0.0f;
  const float alpha      = 1.0f;
  const float beta       = 0.0f;

  float* output = new float[matrix.rows];
  int*   result = new int[matrix.rows];
  std::memset(output, 0, matrix.rows * sizeof(float));
  std::memset(result, 0, matrix.rows * sizeof(int));
#pragma omp parallel for
  for (int i = 0; i < matrix.B_entry[source]; i++) {
    output[matrix.B[source][i]] = 1.0f;
    result[matrix.B[source][i]] = 1;
  }
  cout << "828" << endl;
  cudaMemcpyAsync(d_input, output, sizeof(float) * matrix.rows,
                  cudaMemcpyHostToDevice, streams);
  cudaMemcpyAsync(d_output, output, sizeof(float) * matrix.rows,
                  cudaMemcpyHostToDevice, streams);

  // timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  while (dim < matrix.dim) {
    dim++;

    // for (int j = 0; j < matrix.rows; j++) {
    //   if (output[j] > 0.0f)
    //     printf("[%d,%d]: %f\n", source, j,
    //     output[j]);
    // }
    if (dim % 2 == 0) {
      // cout << "if dim: " << dim << endl;
      cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
                   &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffers);

      cudaMemcpyAsync(output, d_output, sizeof(float) * matrix.rows,
                      cudaMemcpyDeviceToHost, streams);
    }
    // } else {
    //   // cout << "else dim: " << dim << endl;
    //   cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
    //   vecY,
    //                &beta, vecX, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
    //                buffers);

    //   cudaMemcpyAsync(output, d_input, sizeof(float) * matrix.rows,
    //                   cudaMemcpyDeviceToHost, streams);
    // }
    // for (int j = 0; j < matrix.rows; j++) {
    //   if (output[j] > 0.0f)
    //     printf("[%d,%d]: %f\n", source, j,
    //     output[j]);
    // }
    // cout << endl;
    int entry = 0;
#pragma omp parallel for
    for (int j = 0; j < matrix.rows; j++) {
      if ((result[j] == 0) && (output[j] > 0.0f) && (j != source)) {
        result[j] = dim;
        ++entry;
      }
    }
    if ((entry > 0) && (entry < matrix.rows - 1)) {
      entry_last += entry;
      if (entry_last >= matrix.rows - 1)
        break;
    } else {
      break;
    }
  }
  cudaEventRecord(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  cout << "890" << endl;
  matrix.entry += entry_last;

  delete[] output;
  output = nullptr;
  // 输出结果
  if ((matrix.prinft) && (source == matrix.source)) {
    outfile(matrix.rows, result, source, output_path);
  }
  delete[] result;
  result = nullptr;

  cudaStreamSynchronize(streams);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed;
}

void DAWN::outfile(int n, int* result, int source, std::string& output_path)
{
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  cout << "Start outfile" << endl;
  for (int j = 0; j < n; j++) {
    if ((source != j) && (result[j] > 0))
      outfile << source << " " << j << " " << result[j] << endl;
  }
  cout << "End outfile" << endl;
  outfile.close();
}

void DAWN::readCRC(DAWN::Matrix& matrix, std::string& input_path)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }

  matrix.A       = new int*[matrix.rows];
  matrix.A_entry = new int[matrix.rows];
#pragma omp parallel for
  for (int i = 0; i < matrix.rows; i++) {
    matrix.A_entry[i] = 0;
  }

  string line;
  int    rows = 0, cols = 0, k = 0;
  while (getline(file, line)) {
    stringstream ss(line);
    ss >> cols;

    if ((cols == 0) || (cols == matrix.rows)) {
      if (cols == 0)
        rows++;
      continue;
    }

    matrix.A_entry[rows] = cols;
    matrix.A[rows]       = new int[cols];
    for (int j = 0; j < cols; j++) {
      ss >> matrix.A[rows][k++];
    }
    rows++;
    k = 0;
  }
}

void DAWN::readRCC(DAWN::Matrix& matrix, std::string& input_path)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  matrix.B       = new int*[matrix.rows];
  matrix.B_entry = new int[matrix.rows];
#pragma omp parallel for
  for (int i = 0; i < matrix.rows; i++) {
    matrix.B_entry[i] = 0;
  }
  string line;
  int    rows = 0, cols = 0, k = 0;
  while (getline(file, line)) {
    stringstream ss(line);
    ss >> cols;
    if ((cols == 0) || (cols == matrix.rows)) {
      if (cols == 0)
        rows++;
      continue;
    }
    matrix.B_entry[rows] = cols;
    matrix.B[rows]       = new int[cols];
    for (int j = 0; j < cols; j++) {
      ss >> matrix.B[rows][k++];
    }
    rows++;
    k = 0;
  }
}

void DAWN::readGraphBig(std::string&  input_path,
                        std::string&  col_input_path,
                        std::string&  row_input_path,
                        DAWN::Matrix& matrix)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }

  std::string line;
  int         rows, cols, nnz, dim;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;

    std::stringstream ss(line);
    ss >> rows >> cols >> nnz >> dim;
    break;
  }
  cout << rows << " " << cols << " " << nnz << " " << dim << endl;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.dim  = dim;

  cout << "readCRC" << endl;
  readCRC(matrix, col_input_path);
  cout << "readRCC" << endl;
  readRCC(matrix, row_input_path);
  matrix.nnz = 0;
  for (int i = 0; i < matrix.rows; i++) {
    matrix.nnz += matrix.A_entry[i];
  }
  cout << "nnz: " << matrix.nnz << endl;
  matrix.nnz = 0;
  for (int i = 0; i < matrix.rows; i++) {
    matrix.nnz += matrix.B_entry[i];
  }
  matrix.entry = matrix.nnz;
  cout << "nnz: " << matrix.nnz << endl;
  cout << "Initialize Input Matrices" << endl;
}

void DAWN::createGraphconvert(std::string&  input_path,
                              DAWN::Matrix& matrix,
                              std::string&  col_output_path,
                              std::string&  row_output_path)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }

  std::string line;
  int         rows, cols, nnz, dim;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;

    std::stringstream ss(line);
    ss >> rows >> cols >> nnz >> dim;
    break;
  }
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.dim  = dim;

  matrix.A       = new int*[matrix.rows];
  matrix.A_entry = new int[matrix.rows];
  matrix.B       = new int*[matrix.rows];
  matrix.B_entry = new int[matrix.rows];
  matrix.entry   = 0;
#pragma omp parallel for
  for (int i = 0; i < matrix.rows; i++) {
    matrix.A_entry[i] = 0;
    matrix.B_entry[i] = 0;
  }

  vector<pair<int, int>> cooMatCol;
  cout << "Read Input Graph" << endl;
  readGraph(input_path, matrix, cooMatCol);

  cout << "Create Transport Graph" << endl;
  vector<pair<int, int>> cooMatRow = cooMatCol;
  sort(cooMatRow.begin(), cooMatRow.end());

  cout << "Create Input Matrices" << endl;
#pragma omp parallel for
  for (int i = 0; i < matrix.rows; i++) {
    matrix.A[i] = new int[matrix.A_entry[i]];
    matrix.B[i] = new int[matrix.B_entry[i]];
  }
  COO2CRCconvert(matrix, cooMatCol, col_output_path);
  COO2RCCconvert(matrix, cooMatRow, row_output_path);
  cout << "Initialize Input Matrices" << endl;
}

void DAWN::COO2CRCconvert(DAWN::Matrix&                matrix,
                          std::vector<pair<int, int>>& cooMatCol,
                          std::string&                 col_output_path)
{
  std::ofstream outfile(col_output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << col_output_path << std::endl;
    return;
  }

  int         col_a = 0;
  int         k     = 0;
  vector<int> tmp;
  tmp.clear();
  float elapsed_time = 0.0;
  while (k < cooMatCol.size()) {
    auto start = std::chrono::high_resolution_clock::now();
    if (matrix.A_entry[col_a] != 0) {
      if (cooMatCol[k].second == col_a) {
        tmp.push_back(cooMatCol[k].first);
        k++;
      } else {
#pragma omp parallel for
        for (int j = 0; j < matrix.A_entry[col_a]; j++) {
          matrix.A[col_a][j] = tmp[j];
        }
        tmp.clear();
        col_a++;
      }
    } else {
      col_a++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    elapsed_time += elapsed.count();
    infoprint(col_a, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
  }
#pragma omp parallel for
  for (int j = 0; j < matrix.A_entry[col_a]; j++) {
    matrix.A[col_a][j] = tmp[j];
  }
  outfile << matrix.rows << " " << matrix.cols << " " << matrix.nnz << " "
          << matrix.dim << " " << endl;
  for (int i = 0; i < matrix.rows; i++) {
    outfile << matrix.A_entry[i] << " ";
    for (int j = 0; j < matrix.A_entry[i]; j++) {
      outfile << matrix.A[i][j] << " ";
    }
    outfile << endl;
  }
}

void DAWN::COO2RCCconvert(DAWN::Matrix&                matrix,
                          std::vector<pair<int, int>>& cooMatRow,
                          std::string&                 row_output_path)
{
  std::ofstream outfile(row_output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << row_output_path << std::endl;
    return;
  }
  cout << "create B" << endl;

  int         row_b = 0;
  int         k     = 0;
  vector<int> tmp;
  tmp.clear();
  float elapsed_time = 0.0;
  while (k < cooMatRow.size()) {
    auto start = std::chrono::high_resolution_clock::now();
    if (matrix.B_entry[row_b] != 0) {
      if (cooMatRow[k].first == row_b) {
        tmp.push_back(cooMatRow[k].second);
        k++;
      } else {
#pragma omp parallel for
        for (int j = 0; j < matrix.B_entry[row_b]; j++) {
          matrix.B[row_b][j] = tmp[j];
        }
        tmp.clear();
        row_b++;
      }
    } else {
      row_b++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    elapsed_time += elapsed.count();
    infoprint(row_b, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
  }
  cout << "compress B" << endl;
#pragma omp parallel for
  for (int j = 0; j < matrix.B_entry[row_b]; j++) {
    matrix.B[row_b][j] = tmp[j];
  }
  outfile << matrix.rows << " " << matrix.cols << " " << matrix.nnz << " "
          << matrix.dim << " " << endl;
  for (int i = 0; i < matrix.rows; i++) {
    outfile << matrix.B_entry[i] << " ";
    for (int j = 0; j < matrix.B_entry[i]; j++) {
      outfile << matrix.B[i][j] << " ";
    }
    outfile << endl;
  }
}

void DAWN::infoprint(int   entry,
                     int   total,
                     int   interval,
                     int   thread,
                     float elapsed_time)
{
  if (entry % (total / interval) == 0) {
    float completion_percentage =
      static_cast<float>(entry * 100.0f) / static_cast<float>(total);
    std::cout << "Progress: " << completion_percentage << "%" << std::endl;
    std::cout << "Elapsed Time :"
              << static_cast<double>(elapsed_time) / (thread * 1000) << " s"
              << std::endl;
  }
}

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

void DAWN::runSsspCpu(DAWN::Matrix& matrix, std::string& output_path)
{
  int source = matrix.source;
  if (matrix.B_entry[source] == 0) {
    cout << "Source is isolated node, please check" << endl;
    exit(0);
  }

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  float elapsed_time = ssspP(matrix, source, output_path);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

void DAWN::runSsspGpu(DAWN::Matrix& matrix, std::string& output_path)
{
  int source = matrix.source;
  if (matrix.B_entry[source] == 0) {
    cout << "Source is isolated node, please check" << endl;
    exit(0);
  }

  // Copy data to device
  int *d_A_entry, *d_A;
  cudaMalloc((void**)&d_A_entry, sizeof(int) * (matrix.rows + 1));
  cudaMalloc((void**)&d_A, sizeof(int) * matrix.nnz);

  cout << matrix.nnz << endl;
  DAWN dawn;
  dawn.COO2CSR(matrix.A, matrix.A_entry, matrix.csr_row_ptr, matrix.csr_col,
               matrix.csr_val, matrix.rows, matrix.nnz);

  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.A_entry[i]; j++) {
      // cout << " matrix.A[i][j]: " << matrix.A[i][j] << endl;
      if (matrix.A[i][j] > matrix.rows - 1)
        printf("matrix.A[%d,%d]=%d \n", i, j, matrix.A[i][j]);
    }
    // cout << endl;
  }
  std::cerr << "Copy Matrix" << std::endl;

  cudaMemcpy(d_A_entry, matrix.csr_row_ptr, (matrix.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, matrix.csr_col, matrix.nnz * sizeof(int),
             cudaMemcpyHostToDevice);

  for (int i = 0; i < 10; i++)
    cout << " matrix.csr_col[i]: " << matrix.csr_col[i] << endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  float elapsed_time =
    ssspGpu(matrix, source, stream, d_A_entry, d_A, output_path);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << " Elapsed time: " << elapsed_time / (matrix.thread * 1000)
            << std::endl;

  cudaStreamDestroy(stream);
  // Free memory on device
  cudaFree(d_A_entry);
  cudaFree(d_A);
}

void DAWN::runSsspGpuSpmv(DAWN::Matrix& matrix, std::string& output_path)
{
  float elapsed_time = 0.0;

  DAWN dawn;
  dawn.COO2CSR(matrix.A, matrix.A_entry, matrix.csr_row_ptr, matrix.csr_col,
               matrix.csr_val, matrix.rows, matrix.nnz);
  std::cerr << "Copy Matrix" << std::endl;

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  int *  dA_csrOffsets, *dA_columns;
  float* dA_values;
  cudaMalloc((void**)&dA_csrOffsets, (matrix.rows + 1) * sizeof(int));
  cudaMalloc((void**)&dA_columns, matrix.nnz * sizeof(int));
  cudaMalloc((void**)&dA_values, matrix.nnz * sizeof(float));

  cudaMemcpy(dA_csrOffsets, matrix.csr_row_ptr, (matrix.rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_columns, matrix.csr_col, matrix.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dA_values, matrix.csr_val, matrix.nnz * sizeof(float),
             cudaMemcpyHostToDevice);

  cusparseSpMatDescr_t matA;
  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, matrix.rows, matrix.cols, matrix.nnz, dA_csrOffsets,
                    dA_columns, dA_values, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseDnVecDescr_t vecX, vecY;

  float* d_input  = new float[matrix.cols];
  float* d_output = new float[matrix.cols];

  void*        buffers;
  size_t       bufferSizes;
  cudaStream_t streams;

  cusparseCreateDnVec(&vecX, matrix.cols, d_input, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, matrix.cols, d_output, CUDA_R_32F);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecY, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizes);
  cudaMalloc(&buffers, bufferSizes);
  cudaStreamCreate(&streams);

  int source = matrix.source;
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;

  if (matrix.B_entry[source] == 0) {
    cout << "Source is isolated node, please check" << endl;
    exit(0);
  }
  elapsed_time += ssspGpuSpmv(matrix, source, output_path, matA, vecX, vecY,
                              d_input, d_output, buffers, streams, handle);

  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end "
               "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  // Output elapsed time and free remaining resources
  std::cout << "Elapsed Time :"
            << static_cast<double>(elapsed_time) / (matrix.thread * 1000)
            << " s" << std::endl;

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
