#include "dawn.hpp"
using namespace std;

__global__ void vecMatOpe(int rows, int *d_A_entry, int *d_A, bool *input, bool *output, int *result, int source, int dim, int *entry);
void runApspGpu(DAWN::Matrix &matrix, std::string &output_path);
float sssp_gpu(DAWN::Matrix &matrix, int source, cudaStream_t streams, int *d_A_entry, int *d_A, int *&result);

void runApspGpu(DAWN::Matrix &matrix, std::string &output_path)
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return;
    }
    float elapsed_time = 0.0;
    int proEntry = 0;

    omp_set_dynamic(true);
    DAWN dawn;

    std::cerr << "分配显存" << std::endl;

    // Copy data to device
    int *d_A_entry, *d_A;
    cudaMalloc((void **)&d_A_entry, sizeof(int) * matrix.rows);
    cudaMalloc((void **)&d_A, sizeof(int) * matrix.nnz);

    cout << matrix.nnz << endl;
    int *h_A = new int[matrix.nnz];
    int tmp = 0;
    for (int i = 0; i < matrix.rows; i++)
    {
        if (matrix.A_entry[i] == 0)
        {
            matrix.A_entry[i] = matrix.A_entry[i - 1];
            continue;
        }
        for (int j = 0; j < matrix.A_entry[i]; j++)
        {
            h_A[tmp] = matrix.A[i][j];
            // cout << h_A[tmp] << endl;
            tmp++;
        }
        matrix.A_entry[i] = tmp;
        // cout << matrix.A_entry[i] << endl;
    }

    std::cerr << "复制矩阵" << std::endl;

    cudaMemcpy(d_A_entry, matrix.A_entry, sizeof(int) * matrix.rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, sizeof(int) * matrix.nnz, cudaMemcpyHostToDevice);

    // for (int i = 0; i < matrix.nnz; i++)
    // {
    //     printf("A[%d] = %d\n", i, h_A[i]); // 打印元素值
    // }

    // Create streams
    cudaStream_t streams[matrix.stream];
    for (int i = 0; i < matrix.stream; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    for (int i = 0; i < matrix.rows; i++)
    {

        int source = i;
        if (matrix.B_entry[i] == 0)
        {
            proEntry++;
            dawn.infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
            continue;
        }
        int cuda_stream = source % matrix.stream;
        float time_tmp = 0.0f;
        int *result = new int[matrix.rows];
        // if (i == 5)
        time_tmp = sssp_gpu(matrix, source, streams[cuda_stream], d_A_entry, d_A, result);
        elapsed_time += time_tmp;
        proEntry++;
        dawn.infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
        // for (int j = 0; j < matrix.rows; j++)
        // {
        //     if (i != j)
        //         outfile << i << " " << j << " " << result[j] << endl;
        // }
        delete[] result;
        result = nullptr;
    }
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // Output elapsed time and free remaining resources
    std::cout << " Elapsed time: " << elapsed_time / (matrix.thread * 1000) << std::endl;

    outfile.close();

    // Synchronize streams
    for (int i = 0; i < matrix.stream; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Free memory on device
    cudaFree(d_A_entry);
    cudaFree(d_A);
}

float sssp_gpu(DAWN::Matrix &matrix, int source, cudaStream_t streams, int *d_A_entry, int *d_A, int *&result)
{
    int dim = 1;
    int entry = matrix.B_entry[source];
    int entry_last = entry;
    bool *output = new bool[matrix.rows];
    bool *input = new bool[matrix.rows];
    omp_set_dynamic(true);
#pragma omp parallel for
    for (int j = 0; j < matrix.rows; j++)
    {
        input[j] = false;
        output[j] = false;
        result[j] = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < matrix.B_entry[source]; i++)
    {
        input[matrix.B[source][i]] = true;
        output[matrix.B[source][i]] = true;
        result[matrix.B[source][i]] = 1;
    }

    bool *d_input, *d_output;
    int *d_result;
    int *d_entry;
    // int *d_dim, *d_source;
    cudaMalloc((void **)&d_input, sizeof(bool) * matrix.cols);
    cudaMalloc((void **)&d_output, sizeof(bool) * matrix.rows);
    cudaMalloc((void **)&d_result, sizeof(int) * matrix.rows);
    cudaMalloc(&d_entry, sizeof(int));
    // cudaMalloc(&d_dim, sizeof(int));
    // cudaMalloc(&d_source, sizeof(int));

    cudaMemcpyAsync(d_input, input, sizeof(bool) * matrix.rows, cudaMemcpyHostToDevice, streams);
    cudaMemcpyAsync(d_output, output, sizeof(bool) * matrix.rows, cudaMemcpyHostToDevice, streams);
    cudaMemcpyAsync(d_entry, &entry, sizeof(int), cudaMemcpyHostToDevice, streams);
    cudaMemcpyAsync(d_result, result, sizeof(int) * matrix.rows, cudaMemcpyHostToDevice, streams);
    // Launch kernel
    int block_size = matrix.block_size;
    int num_blocks = (matrix.cols + block_size - 1) / block_size;

    auto start = std::chrono::high_resolution_clock::now();
    while (dim < matrix.dim)
    {
        dim++;
        // cout << endl;
        // cout << "计算前：" << endl;
        // cout << "input:" << endl;
        // for (int j = 0; j < matrix.rows; j++)
        // {
        //     cout << input[j] << " ";
        // }
        // cout << "output:" << endl;
        // for (int j = 0; j < matrix.rows; j++)
        // {
        //     cout << output[j] << " ";
        // }
        // cout << endl;

        vecMatOpe<<<num_blocks, block_size, 0, streams>>>(matrix.rows, d_A_entry, d_A, d_input, d_output, d_result, source, dim, d_entry);
        // vecMatOpe(int rows, int *d_A_entry, int *d_A, bool *input, bool *output, int *result, int source, int dim, int *entry);

        // cout << "计算后：" << endl;
        // cout << "input:" << endl;
        // for (int j = 0; j < matrix.rows; j++)
        // {
        //     cout << input[j] << " ";
        // }
        // cout << "output:" << endl;
        // for (int j = 0; j < matrix.rows; j++)
        // {
        //     cout << output[j] << " ";
        // }
        // cout << endl;
        // #pragma omp parallel for
        //         for (int j = 0; j < matrix.rows; j++)
        //         {
        //             if ((result[j] == 0) && (output[j] == true) && (source != j))
        //             {
        //                 result[j] = dim;
        // #pragma omp critical
        //                 {
        //                     entry++;
        //                 }
        //             }
        //             input[j] = output[j];
        //             output[j] = false;
        //         }
        cudaMemcpyAsync(&entry, d_entry, sizeof(int), cudaMemcpyDeviceToHost, streams);

        if ((entry > entry_last) && (entry < matrix.rows))
        {
            entry_last = entry;
            if (entry_last >= matrix.rows - 1) // entry = matrix.rows - 1意味着向量填满，无下一轮
                break;
        }
        else // 如果没有新的最短路径产生，则退出循环
        {
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    cudaMemcpyAsync(result, d_result, sizeof(int) * matrix.rows, cudaMemcpyDeviceToHost, streams);
    matrix.entry += entry_last;

    delete[] output;
    output = nullptr;
    delete[] input;
    input = nullptr;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_result);
    cudaFree(d_entry);

    return elapsed.count();
}

__global__ void vecMatOpe(int rows, int *d_A_entry, int *d_A, bool *input, bool *output, int *result, int source, int dim, int *entry)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // if (j < rows)
    // {
    //     printf("input[%d] = %d\n", j, input[j]);
    // }
    // if (j < rows)
    // {
    //     printf("output[%d] = %d\n", j, output[j]);
    // }

    if (j < rows && (d_A_entry[j] != d_A_entry[j - 1]))
    {

        int start = (j == 0 ? 0 : d_A_entry[j - 1]); // 当前行的起始位置
        int end = d_A_entry[j];                      // 当前行的结束位置
        for (int k = start; k < end; k++)
        { // 索引矩阵A的当前行
          // printf("A[%d] = %d\n", k, d_A[k]); // 打印元素值
            if (input[d_A[k]] == true)
            {
                // printf("A[%d] = %d\n", k, d_A[k]);
                // printf("input[%d] = %d\n", d_A[k], input[d_A[k]]);
                output[j] = true;
                break;
            }
        }
        // printf("A_entry[%d] = %d\n", j, d_A_entry[j]); // 打印元素值
    }
    if (j < rows)
    {
        if ((result[j] == 0) && (output[j] == true) && (source != j))
        {
            result[j] = dim;
            atomicAdd(entry, 1); // use atomic add to ensure entry is incremented safely
        }
        input[j] = output[j];
        output[j] = false;
    }
    // if (j < rows)
    // {
    //     printf("input[%d] = %d\n", j, input[j]);
    // }
    // if (j < rows)
    // {
    //     printf("output[%d] = %d\n", j, output[j]);
    // }
}

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string output_path = argv[2];
    int stream = atoi(argv[3]);
    int block_size = atoi(argv[4]);

    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }

    DAWN dawn;
    DAWN::Matrix matrix;
    matrix.thread = 1; // 运行SSSP的线程,GPU版本默認版本
    matrix.interval = 100;
    matrix.stream = stream;         // 32
    matrix.block_size = block_size; // 4
    dawn.createGraph(input_path, matrix);
    runApspGpu(matrix, output_path);

    return 0;
}