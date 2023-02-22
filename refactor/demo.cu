#include "access.h"
using namespace std;

const int kmaxRow = 10000;
const int kmaxBitsetThou = 1000;
const int kmaxBitsetTenThou = 10000;
const int kmaxBitsetHunThou = 100000;

enum AlgoType {dawn, dij, spfa};

__global__ void denseIntMultiplicationKernel(long long N, long long M, long long K, unsigned int* A, unsigned int* B, bool* res);
void change(long long N, long long M, long long K, bool C[], unsigned int* A);

class Matrix {
public:
	Matrix() {}
	Matrix(string _g_type, string _device, AlgoType _algo_type, int _row, int _col):
			g_type(_g_type), device(_device), algo_type(_algo_type), row(_row), col(_col) {
		if (device == "gpu" && g_type == "dense") {
			stride = (col + 31) / 32;
      		g_dense_gpu = new unsigned int[row * stride]();
		}
	}
	~Matrix() {
		if (device == "gpu" && g_type == "dense") {
			delete[] g_dense_gpu;
		}
	}
	inline void setValue(int i, int j, int value) {
		if (device == "gpu" && g_type == "dense") {
			g_dense_gpu[i * stride + j] = value;
		}
	}
	inline int getValue(int i, int j, bool flag_press) {
		if (device == "gpu" && g_type == "dense") {
			if (flag_press) {
				unsigned int index = i * stride + (j >> 5);
				int len = col % 32;
				int offset = len - (j & 31) - 1;
				// cout << "len:" << len << "  j:" << j << "  offset:" << offset << '\n';
				bool value = (g_dense_gpu[index] & (1 << offset));
				return value;
			} else {
				return g_dense_gpu[i * stride + j];
			}
		}
		return 1;//
	}
	void readData(bool* in) {
		if (device == "gpu" && g_type == "dense") {
			for (int i = 0; i < row; ++i) {
				for (int j = 0; j < stride; ++j) {
					if (in[i * stride + j]) {
						g_dense_gpu[i * stride + j] = 1;
					} else {
						g_dense_gpu[i * stride + j] = 0;
					}
				}
			}
		}
	}
	void pressData(bool *in) {
		if (device == "gpu" && g_type == "dense") {
			for (int i = 0; i < row; ++i) {
				unsigned int tmp = 0;
				for (int j = 0; j < col; ++j) {
					tmp = (tmp << 1) | in[i * col + j];
					if (j & 31 == 31 || j == col - 1) {
						g_dense_gpu[i * stride + (j >> 5)] = tmp;
					}
				}
			}
		}
	}
	void sparseMultiplication(Matrix* A) { // this = this * A
		if (device == "gpu" && g_type == "dense") {
			unsigned int* start_A = this->g_dense_gpu;
			unsigned int* start_B = A->g_dense_gpu;
			unsigned int* h_res = new unsigned int[this->row * this->col]();
			unsigned int* start_res = h_res;
			long long rows = 1;
			for (int i = 0; i < this->row; ++i) { 
				long long r_B = rows;
				bool* h_C = new bool[this->col]();
				bool* start_C = h_C;
				for (int j = 0; j < this->col; j += rows) { 
					r_B = min(rows, this->col - j);
					unsigned int* ptr_A;
					unsigned int* ptr_B;
					bool* ptr_C;

					cudaMalloc((void**)&ptr_A, 1 * this->stride * sizeof(unsigned int));
					cudaMalloc((void**)&ptr_B, r_B * this->stride * sizeof(unsigned int));
					cudaMalloc((void**)&ptr_C, this->row * r_B * sizeof(bool));

					cudaMemcpy(ptr_A, this->g_dense_gpu, 1 * this->stride * sizeof(unsigned int), cudaMemcpyHostToDevice);
					cudaMemcpy(ptr_B, A->g_dense_gpu, r_B * this->stride * sizeof(unsigned int), cudaMemcpyHostToDevice);
					cudaMemset(ptr_C, 0, 1 * r_B * sizeof(bool));
					
					dim3 block(32, 32, 1);
					dim3 grid((this->row + block.x - 1) / block.x, (this->col + block.y - 1) / block.y, 1);

					denseIntMultiplicationKernel<<<grid, block>>>(1, this->stride, r_B, ptr_A, ptr_B, ptr_C);
					cudaDeviceSynchronize();
					cudaError_t error = cudaGetLastError();
					if (error != cudaSuccess) {
						printf("i:%d j:%d  CUDA error: %s\n", i, j, cudaGetErrorString(error));
					}

					cudaMemcpy(h_C, ptr_C, r_B * sizeof(bool), cudaMemcpyDeviceToHost); 
					
					cudaFree(ptr_A);
					cudaFree(ptr_B); 
					cudaFree(ptr_C); 

					A->g_dense_gpu += r_B * this->stride;
					h_C += r_B;
				}
				h_C = start_C;
				change(1, this->stride, this->col, h_C, h_res);
				this->g_dense_gpu += this->stride;
				A->g_dense_gpu = start_B;
				h_res += this->stride;
				delete []h_C;
			}
			this->g_dense_gpu = start_A;
			A->g_dense_gpu = start_B;
			h_res = start_res;
			this->g_dense_gpu = h_res;
			delete []h_res;
		}
	}

	string g_type, device;
	AlgoType algo_type;
	long long row, col;
	vector<pair<int, long long>> g_sparse_cpu[kmaxRow];
	vector<bitset<kmaxBitsetThou>> g_dense_cpu_thou;
	vector<bitset<kmaxBitsetTenThou>> g_dense_cpu_ten_thou;
	vector<bitset<kmaxBitsetHunThou>> g_dense_cpu_hun_thou;
	long long stride;
	unsigned int *g_dense_gpu;
};

class Graph {
public: 
	Graph(string _g_type, string _device, int _node_num, AlgoType _algo_type):
		node_num(_node_num), algo_type(_algo_type) {
		k = 0;
		k_max = 0;
		k_diameter = node_num;
		A = new Matrix(_g_type, _device, algo_type, _node_num, _node_num);
		B = new Matrix(_g_type, _device, algo_type, _node_num, _node_num);
		res = new Matrix("dense", "gpu", algo_type, _node_num, _node_num * 32); // save res with one_dim ptr
	}
	~Graph() {
		delete A;
		delete B;
	}
	static AlgoType parseAlgo(string file_name) {
		AlgoType ans = dawn; //
		return ans; //
	}
	void readMap(string file_name, string random_flag) {
		bool* tmp = new bool[node_num * node_num]();
		bool* transpose_tmp = new bool[node_num * node_num]();
		if (random_flag == "true") {
			ofstream out;
			out.open(file_name);
			for (int i = 0; i < node_num; ++i) {
				for (int j = 0; j < node_num; j += 32) {
					long long ra = rand();
					long long rb = rand();
					long long rn = (ra * RAND_MAX + rb) % INT_MAX;
					for (int k = 0; k < 32 && j + k < node_num; ++k, rn >>= 1) {
						if (i == j + k) continue;
						if (!tmp[i * node_num + j + k]) ++(this->k);
						tmp[i * node_num + j + k] = rn & 1;
						transpose_tmp[(j + k) * node_num + i] = rn & 1;
						if (rn & 1) {
							out << i << ' ' << j + k << '\n';
						}
					}
				}
			}
			out.close();
		} else {
			ifstream in;
			in.open(file_name);
			if (!in) {
				cout << " == File ifstream error: " << file_name << " ==";
			}
			while (in) {
				if (in.eof()) break;
				int a, b; 
				in >> a >> b;
				if (!tmp[a * node_num + b]) ++(this->k);
				tmp[a * node_num + b] = 1;
				transpose_tmp[b * node_num + a] = 1;
			}
			in.close();
		}
		A->pressData(transpose_tmp);
		B->pressData(tmp);
		for (int i = 0; i < node_num; ++i) {
			for (int j = 0; j < node_num; ++j) {
				cout << tmp[i * node_num + j] << ' ';
			}
			cout << '\n';
		}
		cout << "----\n";
		res->readData(tmp);
		delete []tmp;
		delete []transpose_tmp;
	}
	void updateShortestPath(int dim) {
// #pragma omp parallel for
		for (int i = 0; i < node_num; ++i) {
			for (int j = 0; j < node_num; ++j) {
				if (i != j && B->getValue(i, j, true) && res->getValue(i, j, false) == 0) {
					res->setValue(i, j, dim);
					++k;
				}
			}
		}
	}
	void runDawn() {
		long long k_last = 0;
    	long long dim = 1;
		while(1) {
			++dim;
			B->sparseMultiplication(A);
			for (int i = 0; i < node_num; ++i) {
				for (int j = 0; j < node_num; ++j) {
					cout << B->getValue(i, j, true) << ' ';
				}
				cout << '\n';
			}
			updateShortestPath(dim);
			cout << "dim:" << dim << "  k:" << k << '\n';
			if (k > k_max - 1) return ;
			if (k_diameter == dim) return ;
			if (k == k_last) return ;
			k_last = k;
		}
	}
	void runDij() {}
	void runSpfa() {}
	void runShortestPath() {
		if (algo_type == dawn) {
			runDawn();
		} else if (algo_type == dij) {
			runDij();
		} else if (algo_type == spfa) {
			runSpfa();
		}
	}
	void saveRes(string file_name) {
		ofstream out;
		out.open(file_name);
		for (int i = 0; i < res->row; ++i) {
			for (int j = 0; j < res->stride; ++j) {
				out << res->g_dense_gpu[i * res->stride + j] << ' ';
			}
			out << '\n';
		}
		out.close();
	}
	AlgoType algo_type;
	Matrix *A, *B, *res;
	long long k, k_max, k_diameter;
	long long node_num;
};

Graph* readNeighboorMatrix(string file_name, 
		string g_type, string device, string random_flag, string _node_num) {
	long long node_num = stoi(_node_num);
	/*...*/ // read row and col from file
	AlgoType algo_type = Graph::parseAlgo(file_name);
	Graph* g = new Graph(g_type, device, node_num, algo_type); 
	g->readMap(file_name, random_flag); // need to read row and col again
	return g;
}

int main(int argc, char *argv[]) {
	srand(time(nullptr));
	string input_path = argv[1];
    string output_path = argv[2];
	string g_type = argv[3]; // sparse or dense
	string device = argv[4]; // cpu or gpu
	string random_flag = argv[5]; // true or false
	string node_num = argv[6]; // int
	Graph* g = readNeighboorMatrix(input_path, g_type, device, random_flag, node_num);
	g->runShortestPath();
	g->saveRes(output_path);
	delete g;
    return 0;
}

__global__ void denseIntMultiplicationKernel(long long N, long long M, long long K, unsigned int* A, unsigned int* B, bool* res) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < K) {
		unsigned int temp = 0;
		for (int k = 0; k < M; ++k) {
			temp |= A[i * M + k] & B[j * M + k];
		}
		res[i * K + j] = temp ? 1 : 0;
	}
}

void change(long long N, long long M, long long K, bool C[], unsigned int* A) {
    for (int i = 0; i < N; ++i) {
        unsigned int tmp = 0;
        for (int j = 0; j < K; ++j) {
            tmp = (tmp << 1) | C[i * K + j];
            if (j & 31 == 31 || j == K - 1) {
                A[i * M + (j >> 5)] = tmp;
            }
        }
    }
}