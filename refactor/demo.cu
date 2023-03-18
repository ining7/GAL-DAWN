#include "access.h"
using namespace std;

const int kmaxRow = 1000000;
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
		} else if (device == "cpu" && g_type == "sparse") {
			stride = col;
		} else if (device == "cpu" && g_type == "dense") {
			g_dense_cpu_hun_thou.resize(row);
		}
	}
	~Matrix() {
		if (device == "gpu" && g_type == "dense") {
			delete[] g_dense_gpu;
		} else if (device == "cpu" && g_type == "sparse") { 
			delete[] g_dense_gpu;
		} else if (device == "cpu" && g_type == "dense") {
			delete[] g_dense_gpu;
		}
	}
	inline void setValueGpuDense(int i, int j, int value) {
		g_dense_gpu[i * stride + j] = value;
	}
	inline void setValueCpuSparse(int i, int j, long long value) {
		g_sparse_cpu[i].emplace_back(make_pair(j, value));
	}
	inline int getValueGpuDense(int i, int j, bool flag_press) { // flag_press = true press
		if (flag_press) {
			unsigned int index = i * stride + (j >> 5);
			int len = col % 32;
			int offset = len - (j & 31) - 1;
			bool value = (g_dense_gpu[index] & (1 << offset));
			return value;
		} else {
			return g_dense_gpu[i * stride + j];
		}
	}
	inline int getValueCpuSparse(int i, int j) {
		return g_dense_gpu[i * stride  + j];
	}
	inline int getValueCpuDense(int i, int j) {
		return g_dense_gpu[i * stride  + j];
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
		} else if (device == "cpu" && g_type == "sparse") {
			for (int i = 0; i < row; ++i) {
				for (int j = 0; j < col; ++j) {
					if (in[i * col + j]) {
						setValueCpuSparse(i, j, 1);
					}
				}
			}
		} else if (device == "cpu" && g_type == "dense") {
			for (int i = 0; i < row; ++i) {
				for (int j = 0; j < col; ++j) {
					if (in[i * col + j]) {
						g_dense_cpu_hun_thou[i].set(j);
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
		} else if (device == "cpu" && g_type == "sparse") {

		}
	}
	void printMatrix() {
		cout << "\ndevice:" << device << "  g_type:" << g_type << '\n';
		if (device == "gpu" && g_type == "dense") {
			for (int i = 0; i < row; ++i) {
				for (int j = 0; j < stride; ++j) {
					cout << g_dense_gpu[i * stride + j] << ' ';
				}
				cout << '\n';
			}
		} else if (device == "cpu" && g_type == "sparse") {
			for (int i = 0; i < row; ++i) {
				for (int k = 0, arn = g_sparse_cpu[i].size(); k < arn; ++k) {
					int j = g_sparse_cpu[i][k].first;
					int v = g_sparse_cpu[i][k].second;
					cout << "i:" << i << " j:" << j << " v:" << v << '\n';
				}
			}
		} else if (device == "cpu" && g_type == "dense") {
			for (int i = 0, size_i = g_dense_cpu_hun_thou.size(); i < size_i; ++i) {
				for (int j = 0; j < size_i; ++j) {
					cout << (g_dense_cpu_hun_thou[i][j] & 1) << ' ';
				}
				cout << '\n';
			}
		}
		cout << '\n';
	}
	void denseCpuMultiplication(Matrix* A) { // this = this * A
		int n = this->row;
		vector<bitset<kmaxBitsetHunThou>> res(n);
	// #pragma omp parallel for
 		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				bool tmp = (g_dense_cpu_hun_thou[i] & A->g_dense_cpu_hun_thou[j]).any();
				if (res[i][j] == false && tmp == true) {
					res[i].set(j);
				}
			}
		}
		for (int i = 0; i < n; ++i) {
			this->g_dense_cpu_hun_thou[i] = res[i];
		}
		// this->g_dense_cpu_hun_thou = res;
	}
	void sparseCpuMultiplication(Matrix* A) {
		int n = this->row;
		Matrix *tmp = new Matrix(this->g_type, this->device, this->algo_type, this->row, this->col);
		vector<long long> res[n];
		vector<int> b_idx_que[n];
	// #pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			auto a_tmp = this->g_sparse_cpu[i];
			bool flag[n] = {};
			b_idx_que[i].resize(n);
			int b_idx_idx[n] = {};
			int cnt = 0;
			for (int j = 0, m = a_tmp.size(); j < m; ++j) {
				int a_idx = a_tmp[j].first;
				auto b_tmp = A->g_sparse_cpu[a_idx];
				for (int k = 0, l = b_tmp.size(); k < l; ++k) {
					int b_idx = b_tmp[k].first;
					if (!flag[b_idx]) {
						flag[b_idx] = true;
						b_idx_que[i][cnt] = b_idx;
						b_idx_idx[b_idx] = cnt;
						++cnt;
					}
				}
			}
			res[i].resize(cnt);
			for (int j = 0, m = a_tmp.size(); j < m; ++j) {
				int a_idx = a_tmp[j].first;
				int a_w = a_tmp[j].second;
				auto b_tmp = A->g_sparse_cpu[a_idx];
				for (int k = 0, l = b_tmp.size(); k < l; ++k) {
					int b_idx = b_tmp[k].first;
					res[i][b_idx_idx[b_idx]] = (long long) res[i][b_idx_idx[b_idx]] + a_w * b_tmp[k].second;
				}
			}
		}
	// #pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			int m = res[i].size();
			for (int j = 0; j < m; ++j) {
				tmp->g_sparse_cpu[i].emplace_back(make_pair(b_idx_que[i][j], res[i][j]));
			}
		}
		for (int i = 0; i < n; ++i) {
			this->g_sparse_cpu[i] = tmp->g_sparse_cpu[i];
		}
	}
	void denseGpuMultiplication(Matrix* A) { // this = this * A
		unsigned int* start_A = this->g_dense_gpu;
		unsigned int* start_B = A->g_dense_gpu;
		unsigned int* h_res = new unsigned int[this->row * this->col]();
		unsigned int* start_res = h_res;
		long long rows = 1; // matrix partitioning
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

				dim3 block(1, 1, 1);
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
		k_max = node_num * (node_num - 1);
		k_diameter = node_num;
		A = new Matrix(_g_type, _device, algo_type, _node_num, _node_num);
		B = new Matrix(_g_type, _device, algo_type, _node_num, _node_num);
		// save res with one_dim ptr
		if (_g_type == "dense" && _device == "gpu") {
			res = new Matrix("dense", "gpu", algo_type, _node_num, _node_num * 32);
		} else if (_g_type == "sparse" && _device == "cpu") {
			res = new Matrix("dense", "gpu", algo_type, _node_num, _node_num * 32);
		} else if (_g_type == "dense" && _device == "cpu") {
			res = new Matrix("dense", "gpu", algo_type, _node_num, _node_num * 32);
		}
	}
	~Graph() {
		delete A;
		delete B;
	}
	static AlgoType parseAlgo(string file_name) {
		AlgoType ans = dawn; //
		return ans; //
	}
	void readMap(string file_name, string device, string g_type, string random_flag) {
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
						if (rn & 1) {
							out << i << ' ' << j + k << '\n';
							if (!tmp[i * node_num + j + k]) ++(this->k);
						}
						tmp[i * node_num + j + k] = rn & 1;
						transpose_tmp[(j + k) * node_num + i] = rn & 1;
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
			// while (in) {
			// 	if (in.eof()) break;
			// 	int a, b;
			// 	in >> a >> b;
			// 	if (!tmp[a * node_num + b]) ++(this->k);
			// 	tmp[a * node_num + b] = 1;
			// 	transpose_tmp[b * node_num + a] = 1;
			// }
			string line;
			getline(in, line);
			while (line[0] == '%') getline(in, line);

			int num_rows, num_cols, num_entries;
			sscanf(line.c_str(), "%d %d %d", &num_rows, &num_cols, &num_entries);

			for (int i = 0; i < num_entries; ++i) {
				int a, b;
				double val;
				in >> a >> b >> val;
				cout << a << "  " << b << '\n';
				if (!tmp[a * node_num + b]) ++(this->k);
				tmp[a * node_num + b] = 1;
				transpose_tmp[b * node_num + a] = 1;
			}

			in.close();
		}
		if (device == "gpu" && g_type == "dense") {
			A->pressData(transpose_tmp);
			B->pressData(tmp);
			res->readData(tmp);
		} else if (device == "cpu" && g_type == "sparse") { //
			A->readData(tmp);
			B->readData(tmp);
			res->readData(tmp);
		} else if (device == "cpu" && g_type == "dense") {
			A->readData(transpose_tmp);
			B->readData(tmp);
			res->readData(tmp);
		}
		delete []tmp;
		delete []transpose_tmp;
	}
	void updateShortestPath(int dim, string g_type, string device) {
		if (device == "gpu" && g_type == "dense") {
			int* cnt = new int[node_num];
// #pragma omp parallel for
			for (int i = 0; i < node_num; ++i) {
				for (int j = 0; j < node_num; ++j) {
					if (i != j && B->getValueGpuDense(i, j, true) && res->getValueGpuDense(i, j, false) == 0) {
						res->setValueGpuDense(i, j, dim);
						++cnt[i];
					}
				}
			}
			for (int i = 0; i < node_num; ++i) {
				k += cnt[i];
			}
		} else if (device == "cpu" && g_type == "sparse") { //
			int* cnt = new int[node_num];
		// #pragma omp parallel for
			for (int i = 0; i < node_num; ++i) {
				cnt[i] = 0;
				auto ed = B->g_sparse_cpu[i].end();
				for (auto it = B->g_sparse_cpu[i].begin(); it != ed; ++it) {
					int j = it->first;
					if (i == j || it->second == 0) continue;
					if (res->getValueCpuSparse(i, j) == 0) {
						res->setValueGpuDense(i, j, dim);
						++cnt[i];
					}
				}
			}
			for (int i = 0; i < node_num; ++i) {
				k += cnt[i];
			}
		} else if (device == "cpu" && g_type == "dense") {
			int* cnt = new int[node_num];
	// #pragma omp parallel for
			for (int i = 0; i < node_num; ++i) {
				cnt[i] = 0;
				for (int j = 0; j < node_num; ++j) {
					if ((i != j) && B->g_dense_cpu_hun_thou[i].test(j) && res->getValueCpuDense(i, j) == 0) {
						res->setValueGpuDense(i, j, dim);
						++cnt[i];
					}
				}
			}
			for (int i = 0; i < node_num; ++i) {
				k += cnt[i];
			}
		}
	}
	void runDawn(string g_type, string device) {
		long long k_last = 0;
    	long long dim = 1;
		while(1) {
			++dim;
			if (g_type == "dense" && device == "gpu") {
				B->denseGpuMultiplication(A) ;
			} else if (g_type == "sparse" && device == "cpu") {
				B->sparseCpuMultiplication(A);
			} else if (g_type == "dense" && device == "cpu") {
				B->denseCpuMultiplication(A);
			}
			updateShortestPath(dim, g_type, device);
			if (k > k_max - 1) return ;
			if (k_diameter == dim) return ;
			if (k == k_last) return ;
			k_last = k;
		}
	}
	void runDij() {}
	void runSpfa() {}
	void runShortestPath(string g_type, string device) {
		if (algo_type == dawn) {
			runDawn(g_type, device);
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
	g->readMap(file_name, device, g_type , random_flag); // need to read row and col again
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
	g->runShortestPath(g_type, device);
	g->saveRes(output_path);
	delete g;
    return 0;
}

__global__ void denseIntMultiplicationKernel(long long N, long long M, long long K, unsigned int* A, unsigned int* B, bool* res) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	// printf("i: %d  j:%d  ", i, j);
	if (i < N && j < K) {
		unsigned int temp = 0;
		for (int k = 0; k < M; ++k) {
			temp |= A[i * M + k] & B[j * M + k];
		}
		// printf("idx: %lld\n", i * K + j);
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