#include <bits/stdc++.h>
using namespace std;

const int kmaxRow = 10000;
const int kmaxBitsetThou = 1000;
const int kmaxBitsetTenThou = 10000;
const int kmaxBitsetHunThou = 100000;

enum AlgoType {dawn, dij, spfa};

class Matrix {
public:
	Matrix(string _g_type, string _device, int _row, int _col):
			g_type(_g_type), device(_device), row(_row), col(_col) {
		stride = col;
		if (device == "gpu" && g_type == "dense") {
			g_dense_gpu = new int[row * col]();
		}
	}
	~Matrix() {
		if (device == "gpu" && g_type == "dense") {
			delete[] g_dense_gpu;
		}
	}
	inline void setValue(int i, int j, int value) {}
	inline int getValue(int i, int j) {}
	string g_type, device;
	int row, col;
	vector<pair<int, long long>> g_sparse_cpu[kmaxRow];
	vector<bitset<kmaxBitsetThou>> g_dense_cpu_thou;
	vector<bitset<kmaxBitsetTenThou>> g_dense_cpu_ten_thou;
	vector<bitset<kmaxBitsetHunThou>> g_dense_cpu_hun_thou;
	int stride;
	int *g_dense_gpu;
}

class Graph {
public:
	Graph(string _g_type, string _device, int _node_num, int _algo_type):
		node_num(_node_num), algo_type(_algo_type) {
		k = 0;
		k_max = 0;
		k_diameter = 0;
		A = new Matrix(g_type, device, _node_num, _node_num);
		B = new Matrix(g_type, device, _node_num, _node_num);
		res = new Matrix("dense", "gpu", _node_num, _node_num); // save res with one_dim ptr
	}
	~Graph() {
		delete A;
		delete B;
	}
	AlgoType parseAlgo(string file_name) {
		// if () {
		// 	return ;
		// }
	}
	void readMap(string file_name) {
		algo_type = parseAlgo(file_name);
	}
	inline void updateShortestPath() {
		#pragma omp parallel for
		for (int i = 0; i < node_num; ++i) {
			for (int j = 0; j < node_num; ++j) {
				if (i != j && B->getValue(i, j) && res->getValue(i, j) == 0) {
					res->setValue(i, j, dim);
					++k;
				}
			}
		}
	}
	inline void sparseMultiplication(Matrix* A, Matrix* B) {}
	void runDawn() {
		long long k_last = 0;
    	long long dim = 1;
		while(1) {
			++dim;
			sparseMultiplication(B, A);
			updateShortestPath();
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
	void saveRes(string file_name) {}
	AlgoType algo_type;
	Matrix* A, B, res;
	long long k, k_max, k_diameter;
	int node_num;
}

void readNeighboorMatrix(string file_name, Graph *g,
		string g_type, string device) {
	int node_num;
	/*...*/ // read row and col from file
	g = new Graph(g_type, device, node_num);
	g->readMap(file_name); // need to read row and col again
}

int main(int argc, char *argv[])
{
	string input_path = argv[1];
    string output_path = argv[2];
	string g_type = argv[3]; // sparse or dense
	string device = argv[4]; // cpu or gpu
	Graph* g;
	readNeighboorMatrix(input_path, g, g_type, device);
	g->runShortestPath();
	g->saveRes(output_path);
	delete g;
    return 0;
}
