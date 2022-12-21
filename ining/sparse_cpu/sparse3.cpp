#include "access.h"

using namespace std;

// Path to the input dataset
string input_path = "";
// The upper limit of the number of nodes
const int NodesUpper = 1e4 + 10;
// setw(stw) to control matrix output
const int stw = 5;

int N; 
vector<vector<int>> shortest_distance(NodesUpper, vector<int>(NodesUpper, 0));

struct Matrix{
    int n; // nodes number
    vector<pair<int, int>> g[NodesUpper];
};

void setElement(Matrix& A, int idx, int jdx, int v);
void printRes();
void sparseMultiplication(Matrix& A, Matrix& B);
void writeList( Matrix& A, Matrix& B, int select_direction);
void DAWN(Matrix& A, Matrix& B, int k, int k_max, int k_diameter);
void unweightedPipeline(int nodes, int select_direction);

void setElement(Matrix& A, int idx, int jdx, int v) {
    A.g[idx].emplace_back(make_pair(jdx, v));
}

void printRes() {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (shortest_distance[i][j]) {
                cout << setw(stw) << shortest_distance[i][j] << ' ';
            } else cout << setw(stw) << 0 << ' ';
        }
        cout << '\n';
    }
}

// A = A * B
void sparseMultiplication(Matrix& A, Matrix& B) {
    int n = A.n;
    Matrix tmp; 
    tmp.n = n; 
    vector<int> res[n]; 
    vector<int> b_idx_que[n]; 
    for (int i = 0; i < n; ++i) {
        auto a_tmp = A.g[i]; 
        bool flag[n]; 
        b_idx_que[i].resize(n); 
        int b_idx_idx[n] = {}; 
        int cnt = 0; 
        for (int j = 0, m = a_tmp.size(); j < m; ++j) { 
            int a_idx = a_tmp[j].first;  
            auto b_tmp = B.g[a_idx];
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
            auto b_tmp = B.g[a_idx];
            for (int k = 0, l = b_tmp.size(); k < l; ++k) {
                int b_idx = b_tmp[k].first; 
                int b_w = b_tmp[k].second; 
                res[i][b_idx_idx[b_idx]] += a_w * b_w; 
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        int m = res[i].size();
        for (int j = 0; j < m; ++j) {
            tmp.g[i].emplace_back(make_pair(b_idx_que[i][j], res[i][j])); 
        }
    }
    A = tmp;
}

void writeList(Matrix& A, Matrix& B, int select_direction) {
    ifstream infile;
    infile.open(input_path);
    if(!infile) cout << " === File error ===\n";
    while(infile) {
        int a, b; 
        infile >> a >> b;
        setElement(A, a, b, 1);
        setElement(B, a, b, 1);
        shortest_distance[a][b] = 1;
        if (select_direction == 1) {
            setElement(A, b, a, 1);
            setElement(B, b, a, 1);
            shortest_distance[b][a] = 1;
        }
    }
    infile.close();
}

void DAWN(Matrix& A, Matrix& B, int k, int k_max, int k_diameter) {
    int k_last = 0;
    int dim = 1;
    int n = N;
    while(1) {
        ++dim;
        sparseMultiplication(B, A);
        for (int i = 0; i < n; ++i) {
            auto ed = B.g[i].end();
            for (auto it = B.g[i].begin(); it != ed; ++it) {
                int j = it->first;
                if (i == j || it->second == 0) continue; 
                if (shortest_distance[i][j] == 0) {
                    shortest_distance[i][j] = dim;
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
    int n = N;
    Matrix A, B;
    A.n = B.n = n;
    int k = 0; // The number of pairs of nodes that have found the shortest path
    long long k_max = 1.0 * nodes * (nodes - 1); // Maximum number of node pairs
    int k_diameter = n; // Graph diameter
    // cout << "Please enter the graph diameter: " << '\n';
    // cin >> k_diameter;
    cout << "[default: graph diameter = nodes number]\n";

    writeList(A, B, select_direction);

    clock_t start, finish;
    // start the timer
    start = clock();
    cout << "Timing begins\n";

    DAWN(A, B, k, k_max, k_diameter);

    finish = clock();
    // stop the timer
    cout << "The total running time of the program is " << double(finish - start) / CLOCKS_PER_SEC << "s\n";
}

int main(int argc, char *argv[]) {
    // get file path
    input_path = argv[1]; 

    cout << "Please enter the number of nodes in the graph: \n";
    cin >> N;
    int nodes = N;

    // odd matrix
    if (N % 2 != 0) N = N + 1;

    int select_weight = 0, select_direction = 0;
    cout << "[default: unweighted graph]\n";
    // cout << "Please select the type of diagram: 1-Unweighted graph  2-Weighted graph\n";
    // cin >> select_weight;
    cout << "Please select the type of diagram: 1-Undirected graph  2-Directed graph\n";
    cin >> select_direction;

    unweightedPipeline(nodes, select_direction);

    // Output the shortest path result
    printRes();

    return 0;
}