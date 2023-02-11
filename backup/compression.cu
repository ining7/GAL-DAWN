#include "access.h"
#include "timer.hpp"

using namespace std;

string output_path = "../compression_out1.txt";
string A_path = "";
string B_path = "";

void writeMatrix_bool_A(long long R, bool*& A) {
    ifstream in;
    in.open(A_path);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < R; ++j) {
            int x; 
            in >> x;
            if (x & 1) {
                A[i * R + j] = 1;
            } else {
                A[i * R + j] = 0;
            }
        }
    }
}

void writeMatrix_bool_B(long long R, bool*& B) {
    ifstream in;
    in.open(B_path);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < R; ++j) {
            int x; 
            in >> x;
            if (x & 1) {
                B[i * R + j] = 1;
            } else {
                B[i * R + j] = 0;
            }
        }
    }
} 

__global__ void denseBoolMultiplicationKernel(long long N, long long M, long long K, unsigned int* A, unsigned int* B, bool* res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;
    if (i < N && j < K) {
        unsigned int temp = 0;
        for (k = 0; k < M; ++k) {
            temp |= A[i * M + k] & B[j * M + k];
        }
        res[i * K + j] = temp ? 1 : 0;
    }
}

__global__ void changeKernel(long long N, long long M, long long K, bool C[], unsigned int* A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < K) {
        int len = (N - j) < 32 ? (N - j - 1) : 31;
        len += j % 32;
        int offset = len - (j % 32);
        // A[i * M + (j >> 5)] += (C[i * N + j] << offset);
        atomicAdd(&A[i * M + (j >> 5)], (C[i * N + j] << offset));
        // printf("i:%d j:%d C:%d len:%d offset:%d   c:%d a:%d\n", i, j, C[i * N + j], len, offset , C[i * N + j] << offset, A[i * M + (j >> 5)]); 
    }
}

void change(long long R, long long M, bool C[], unsigned int* A) {
    for (int i = 0; i < R; ++i) {
        unsigned int tmp = 0;
        for (int j = 0; j < R; ++j) {
            tmp = (tmp << 1) | C[i * R + j];
            if (j & 31 == 31 || j == R - 1) {
                A[i * M + (j >> 5)] = tmp;
                // cout << "i:" << i << "  (j>>5):" << (j >> 5) << "  idx:" << (i * M + (j >> 5)) << "  tmp:" << tmp << '\n';
            }
        }
    }
}

// int blockSize;
// cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0);
// printf("Max threads per block: %d\n", blockSize);
int main(int argc, char *argv[]) {
    string A_path = argv[1];
    string B_path = argv[2];

    const long long R = 50000;
    const long long N = R;
    const long long M = (R + 31) / 32;
    const long long K = R;
    bool* h_A_bool = new bool[N * N]();
    bool* h_B_bool = new bool[N * N]();
    unsigned int* h_res = new unsigned int[N * M]();

    writeMatrix_bool_A(N, h_A_bool);
    writeMatrix_bool_B(N, h_B_bool);

    unsigned int* h_A = new unsigned int[N * M]();
    unsigned int* h_B = new unsigned int[K * M]();

    change(N, M, h_A_bool, h_A);
    change(N, M, h_B_bool, h_B);

    unsigned int* A;
    unsigned int* B;
    bool* C;
    unsigned int* res;

    cudaMalloc((void**)&A, N * M * sizeof(unsigned int));
    cudaMalloc((void**)&B, M * K * sizeof(unsigned int));
    cudaMalloc((void**)&C, N * K * sizeof(bool));
    cudaMalloc((void**)&res, N * M * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    cudaMemcpy(A, h_A, N * M * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, K * M * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(C, 0, N * K * sizeof(bool));
    cudaMemset(res, 0, N * M * sizeof(unsigned int));
    
    dim3 block(32, 32, 1);
    dim3 grid((N + block.x - 1) / block.x, (K + block.y - 1) / block.y, 1);
    
    denseBoolMultiplicationKernel<<<grid, block>>>(N, M, K, A, B, C);
    cudaDeviceSynchronize(); 

    changeKernel<<<grid, block>>>(N, M, K, C, res);
    cudaDeviceSynchronize();
// cudaEventRecord(start);
    cudaMemcpy(h_res, res, N * M * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "time: " << elapsedTime << "ms" << endl;
    
    cudaFree(A);
    cudaFree(B); 

    // ofstream out;
    // out.open(output_path);
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < M; ++j) {
    //         out << h_res[i * M + j] << ' ';
    //     }
    //     out << '\n';
    // }
    // out.close();

    return 0;
}