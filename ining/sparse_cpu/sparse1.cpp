#include "access.h"

using namespace std;

// Path to the input dataset
string input_path = "";
// The upper limit of the number of nodes
const int NodesUpper = 1e3;
// setw(stw) to control matrix output
const int stw = 5;

class Matrix {
private:
    int N;
    vector<pair<int, int>> g[NodesUpper];

public:
    Matrix(int x) : N(x) {}

    void setElement(int idx, int jdx, int v) {
        g[idx].emplace_back(make_pair(jdx, v));
    }

    void changeElement(int idx, int jdx, int v) {
        int n = g[idx].size();
        for (int i = 0; i < n; ++i) {
            if (g[idx][i].first == jdx) {
                g[idx][i].second = v;
                return ;
            }
        }
        setElement(idx, jdx, v);
    }

    int getElement(int idx, int jdx) {
        int n = g[idx].size();
        for (int i = 0; i < n; ++i) {
            if (g[idx][i].first == jdx) {
                return g[idx][i].second;
            }
        }
        return 0;
    }

    int getMatrixSize() {
        int cnt = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (getElement(i, j)) ++cnt;
            }
        }
        return cnt;
    }

    void printMatrix() {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << setw(stw) << this->getElement(i, j) << ' ';
            }
            cout << '\n';
        }
    }

    // A = A * B;
    void sparseMultiplication(Matrix C) {
        map<int, int> res[N];
        auto B = C.g;
        for (int i = 0; i < N; ++i) {
            auto a_tmp = this->g[i];
            map<int, int> res_tmp;
            for (int j = 0, n = a_tmp.size(); j < n; ++j) {
                int a_idx = a_tmp[j].first;
                int a_w = a_tmp[j].second;
                auto b_tmp = B[a_idx];
                for (int k = 0, m = b_tmp.size(); k < m; ++k) {
                    int b_idx = b_tmp[k].first;
                    int b_w = b_tmp[k].second;
                    res_tmp[b_idx] += a_w * b_w; 
                }
            }
            res[i] = res_tmp;
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (res[i][j]) this->changeElement(i, j, res[i][j]);
                else this->changeElement(i, j, 0);
            }
        }
    }

    void readList(Matrix& A, Matrix& B, int select_direction) {
        ifstream infile;
        infile.open(input_path);
        if(!infile) cout << " === File error ===\n";
        while(infile) {
            int a, b; 
            infile >> a >> b;
            A.setElement(a, b, 1);
            B.setElement(a, b, 1);
            this->setElement(a, b, 1);
            if (select_direction == 1) {
                A.setElement(b, a, 1);
                B.setElement(b, a, 1);
                this->setElement(b, a, 1);
            }
        }
        infile.close();
    }

    void DAWN(Matrix& A, Matrix& B, int k, int k_max, int k_diameter) {
        int k_last = 0;
        int dim = 1;
        while(1) {
            ++dim;
            B.sparseMultiplication(A);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (B.getElement(i, j) != 0 && this->getElement(i, j) == 0 && i != j) {
                        this->changeElement(i, j, dim);
                        ++k;
                        if (k > k_max - 1) return ;
                    }
                }   
                if (k > k_max - 1) return ;
            }
            if (k > k_max - 1) return ;
            if (k_diameter == dim) return ;
            k_last = k;
        }
    }

    void unweightedPipeline(int nodes, int select_direction) {
        // init
        Matrix A(this->N), B(this->N);
        int k = 0; // The number of pairs of nodes that have found the shortest path
        long long k_max = 1.0 * nodes * (nodes - 1); // Maximum number of node pairs
        int k_diameter = this->N; // Graph diameter
        // cout << "Please enter the graph diameter: " << '\n';
        // cin >> k_diameter;
        cout << "[default: graph diameter = nodes number]\n";

        this->readList(A, B, select_direction);
        
        clock_t start, finish;
        // start the timer
        start = clock();
        cout << "Timing begins\n";

        this->DAWN(A, B, k, k_max, k_diameter);

        finish = clock();
        // stop the timer
        cout << "The total running time of the program is " << double(finish - start) / CLOCKS_PER_SEC << "s\n";
    }

};

int main(int argc, char *argv[]) {
    // get file path
    input_path = argv[1];

    int N = 0;
    cout << "Please enter the number of nodes in the graph: \n";
    cin >> N;
    int nodes = N;

    // odd matrix
    if (N % 2 != 0) N = N + 1;

    Matrix shortest_distance(N);
 
    int select_weight = 0, select_direction = 0;
    cout << "[default: unweighted graph]\n";
    // cout << "Please select the type of diagram: 1-Unweighted graph  2-Weighted graph\n";
    // cin >> select_weight;
    cout << "Please select the type of diagram: 1-Undirected graph  2-Directed graph\n";
    cin >> select_direction;

    shortest_distance.unweightedPipeline(nodes, select_direction); 
     
    // Output the shortest path result
    shortest_distance.printMatrix();

    return 0;
}