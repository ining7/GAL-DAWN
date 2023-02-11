#include "access.h"
#include "timer.hpp"
#include "omp.h"
#include <immintrin.h>

using namespace std;

const long long R = 10000;
const int N = R;
const int M = (R + 31) / 32;
const int K = R;

string output_path = "../avx_out.txt";

void writeMatrix_bool_A(bool*& A) {
    ifstream in;
    in.open("../A.txt");
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < R; ++j) {
            int x; 
            in >> x;
            if (x == 1) {
                A[i * R + j] = 1;
            }
        }
    }
}

void writeMatrix_bool_B(bool*& B) {
    ifstream in;
    in.open("../B.txt");
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < R; ++j) {
            int x; 
            in >> x;
            if (x == 1) {
                B[j * R + i] = 1;
            }
        }
    }
}  

void denseMultiplication_int(unsigned int *A, unsigned int *B, unsigned int *res) {
    unsigned int *A_start = A, *B_start = B, *res_start = res;
    unsigned int *A_end = A + N * M, *B_end = B + M * K, *res_end = res + N * K;
    int avx_len = M / 8;
    int avx_remain = M % 8;
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        B = B_start;
        for (int j = 0; j < K; j++) {
            A = A_start + i * M;
            unsigned int tmp = 0;
            for (int k = 0; k < avx_len; k++) {
                __m256i a = _mm256_loadu_si256((__m256i*)A);
                __m256i b = _mm256_loadu_si256((__m256i*)B);
                __m256i c = _mm256_and_si256(a, b);
                A += 8;
                B += 8;
                tmp |= _mm256_movemask_ps((__m256)c);
            }
            for (int k = 0; k < avx_remain; k++) {
                tmp |= (*A) & (*B);
                A++;
                B++;
            }
            *(res + i * K + j) = tmp;
        }
    }
    res = res_start;
}

void change(bool C[], unsigned int*& A) {
    for (int i = 0; i < R; ++i) {
        unsigned int tmp = 0;
        for (int j = 0; j < R; ++j) {
            tmp = (tmp << 1) | C[i * R + j];
            if (j & 31 == 31 || j == R - 1) {
                A[i * M + (j >> 5)] = tmp;
            }
        }
    }
}

int main(int argc, char *argv[]) {

    bool* C = new bool[R * R]();
    bool* D = new bool[R * R]();

    writeMatrix_bool_A(C);
    writeMatrix_bool_B(D);

    unsigned int* A = new unsigned int[N * M]();
    unsigned int* B = new unsigned int[M * K]();

    change(C, A);
    change(D, B);

    unsigned int* res = new unsigned int[N * K]();

    {
        Timer mulTime("avx");
        denseMultiplication_int(A, B, res);
    }

    ofstream out;
        out.open(output_path);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < K; ++j) {
                if (res[i * K + j]) {
                    out << 1 << ' ';
                } else {
                    out << 0 << ' ';
                }
            }
            out << '\n';
        }
    out.close();

    return 0;
}