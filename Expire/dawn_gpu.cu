#include "access.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "omp.h"

using namespace std;

struct Matrix
{
    int rows;
    int cols;
    int nnz;
    int **dense;
    int *dense_entry;
    int m;
    int n;
    int k;
    int loop;
    bool **result;
    int dim;
};

float runDawnGpu(Matrix &matrix, string &input_path, string &output_path);
void readgraph(string &input_path, Matrix &matrix);
void update_A(__half *&host, Matrix matrix, int rows_start, int rows_end, int n);
void update_B(__half *&host, Matrix matrix, int cols_start, int cols_end, int n);
void check_gpu_dense(__half *dense, int i_dex, int j_dex, string &output_path);

void readgraph(string &input_path, Matrix &matrix)
{
    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }

    std::string line;
    int rows, cols, nnz, dim;
    while (std::getline(file, line))
    {
        if (line[0] == '%')
            continue;

        std::stringstream ss(line);
        ss >> rows >> cols >> nnz >> dim;
        break;
    }
    cout << rows << " " << cols << " " << nnz << " " << dim << endl;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;
    matrix.m = 8192;
    matrix.n = (rows - rows % matrix.m) + matrix.m;
    matrix.k = matrix.m;
    matrix.loop = matrix.n / matrix.m;
    matrix.dim = dim;

    matrix.result = new bool *[matrix.n];
    matrix.dense = new int *[matrix.n];
    matrix.dense_entry = new int[matrix.n];
#pragma omp parallel for
    for (int i = 0; i < matrix.n; i++)
    {
        matrix.result[i] = new bool[matrix.n];
        matrix.dense_entry[i] = 0;
        for (int j = 0; j < matrix.n; j++)
        {
            matrix.result[i][j] = false;
        }
    }
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        ss >> rows >> cols;
        rows--;
        cols--;
        if (rows != cols)
            matrix.result[rows][cols] = true;
    }
    file.close();

#pragma omp parallel for
    for (int i = 0; i < matrix.n; i++)
    {
        vector<int> tmp;
        tmp.clear();
        int entry_tmp = 0;
        for (int j = 0; j < matrix.n; j++)
        {
            if (matrix.result[j][i] == true && i != j)
            {
                tmp.push_back(j);
                entry_tmp++;
            }
        }
        matrix.dense_entry[i] = entry_tmp;
        matrix.dense[i] = new int[entry_tmp];
        for (int j = 0; j < entry_tmp; j++)
        {
            matrix.dense[i][j] = tmp[j];
        }
    }
    cout << "Initialize input matrices" << endl;
}

float runDawnGpu(Matrix &matrix, string &input_path, string &output_path)
{
    readgraph(input_path, matrix);
    int n = matrix.n;
    int m = matrix.m;
    int k = matrix.k;
    int loop = matrix.loop;

    cout << " m: " << m << " n: " << n << " k: " << k << " loop: " << matrix.loop << endl;

    int lda = m;
    int ldb = n;
    int ldc = m;

    __half *h_A = new __half[lda * n]; // m*n
    __half *h_B = new __half[ldb * k]; // n*k
    __half *h_C = new __half[ldc * k]; // m*k

    cout << " Host memory allocated " << endl;

    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, lda * n * sizeof(__half));
    cudaMalloc(&d_B, ldb * k * sizeof(__half));
    cudaMalloc(&d_C, ldc * k * sizeof(__half));

    cublasHandle_t handle;
    cublasCreate(&handle);

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    float elapsed_time = 0.0f;
    uint64_t entry = 0;
    uint64_t entry_last = 0;
    int dim = 1; // dim from 1

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize input matrices

    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return 0.0f;
    }
    int edges = 0;

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (matrix.result[i][j] == true && i != j)
            {
#pragma omp atomic
                ++edges;
                // outfile << i << " " << j << " " << dim << endl;
            }
        }
    }
    cout << "Path length 1:" << edges << endl;
    std::cout
        << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
        << std::endl;
    while (dim < matrix.dim)
    {

        dim++;
        // Main loop
        for (int i_std = 0; i_std < loop; i_std++)
        {
            cout << " i: " << i_std << " ";
            update_A(h_A, matrix, i_std * m, (i_std + 1) * m - 1, n);
            // Copy data to device
            cudaMemcpy(d_A, h_A, lda * n * sizeof(__half), cudaMemcpyHostToDevice);

            for (int j_std = 0; j_std < loop; j_std++)
            {
                update_B(h_B, matrix, j_std * k, (j_std + 1) * k - 1, n);
                cudaMemcpy(d_B, h_B, ldb * k * sizeof(__half), cudaMemcpyHostToDevice);

                cudaEventRecord(start, 0);

                // Perform matrix multiplication
                cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             k,
                             n,
                             &alpha,
                             d_A,
                             CUDA_R_16F,
                             lda,
                             d_B,
                             CUDA_R_16F,
                             ldb,
                             &beta,
                             d_C,
                             CUDA_R_16F,
                             ldc,
                             CUBLAS_COMPUTE_16F,
                             CUBLAS_GEMM_DEFAULT);

                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float trial_time;
                cudaEventElapsedTime(&trial_time, start, stop);
                elapsed_time += trial_time;

                cudaMemcpy(h_C, d_C, ldc * k * sizeof(__half), cudaMemcpyDeviceToHost);

                // update result
                for (int i = i_std * m; i < (i_std + 1) * m - 1; i++)
                {
                    for (int j = j_std * k; j < (j_std + 1) * k - 1; j++)
                    {
                        float tmp = h_C[(i % m) * k + (j % k)];
                        if ((tmp != 0) && (matrix.result[i][j] == false))
                        {
                            matrix.result[i][j] = true;
                            //                             if (i != j)
                            //                             {
                            // #pragma omp atomic
                            //                                 ++entry;
                            // outfile << i << " " << j << " " << dim << endl;
                            // }
                        }
                    }
                }
            }
            cout << endl;
        }
        std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;

        cout << "dim : " << dim << endl;
        cout << "entry : " << entry << endl;

        if (entry > matrix.rows * (matrix.rows - 1) - 1)
            break;

        if (entry > entry_last)
        {
            entry_last = entry;
        }
        else
        {
            break;
        }
    }
    std::cout
        << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
        << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy handle
    cublasDestroy(handle);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    outfile.close();

    return elapsed_time;
}

void update_A(__half *&host, Matrix matrix, int rows_start, int rows_end, int n) // m*n
{
#pragma omp parallel for
    for (int i = rows_start; i <= rows_end; i++)
    {
        for (int j = 0; j <= n - 1; j++)
        {
            if (matrix.result[i][j] == true)
            {
                host[(i % matrix.m) * n + j] = __float2half(1.0f);
                // #pragma omp atomic
                //                 ++entry;
            }
            else
                host[(i % matrix.m) * n + j] = __float2half(0.0f);
        }
    }
}

void update_B(__half *&host, Matrix matrix, int cols_start, int cols_end, int n) // n*k
{
    // int entry = 0;
#pragma omp parallel for
    for (int j = cols_start; j <= cols_end; j++)
    {
        for (int i = 0; i < n; i++)
        {
            host[i * matrix.k + j % matrix.k] = __float2half(0.0f);
        }
        for (int i = 0; i < matrix.dense_entry[j]; i++)
        {
            host[matrix.dense[j][i] * matrix.k + j % matrix.k] = __float2half(1.0f);
            // #pragma omp atomic
            //             ++entry;
        }
    }
}

void check_gpu_dense(__half *dense, int i_dex, int j_dex, string &output_path)
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return;
    }
    int entry = 0;

    for (auto i = 0; i < i_dex; i++)
    {
        for (auto j = 0; j < i_dex; j++)
        {
            float tmp = dense[i * j_dex + j];
            if (tmp > 0)
            {
                outfile << i << " " << j << " " << tmp << endl;
                entry++;
            }
        }
    }
    outfile.close();
}

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string output_path = argv[2];

    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }

    Matrix matrix;

    float elapsed_time = 1.0f * runDawnGpu(matrix, input_path, output_path);

    // Output elapsed time and free remaining resources
    std::cout << "Average Elapsed time: " << elapsed_time / (1000 * matrix.loop) << std::endl;
    std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;

    return 0;
}
