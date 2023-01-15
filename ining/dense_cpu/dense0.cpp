// init version | without any optimization
#include "access.h"
#include "timer.hpp"

using namespace std;

// Path to the input dataset
string input_path = "";
string out_path = "";

int N = 0;

void printRes(long long**& shortest_distance) {
    ofstream out(out_path);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out << setw(4) << shortest_distance[i][j] << (j == N - 1 ? '\n' : ' ');
        }
    }
    out.close();
}

void writeList(long long**& shortest_distance, bool**& A, bool**& B, int select_direction) {
    ifstream infile;
    infile.open(input_path);
    if(!infile) cout << " === File error ===\n";
    while(infile) {
        int a, b; 
        infile >> a >> b;
        A[a][b] = true;
        B[a][b] = true;
        shortest_distance[a][b] = 1;
        if (select_direction == 1) {
            A[b][a] = true;
            B[b][a] = true;
            shortest_distance[b][a] = 1;
        }
    }
    infile.close();
}

void sparseMultiplication(bool**& A, bool**& B) {
    bool** res = new bool*[N];
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        res[i] = new bool[N]();
    }
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                res[i][j] = res[i][j] || (A[i][k] && B[k][j]);
            }
        }
    }
    A = res;
}

void DAWN(long long**& shortest_distance, bool**&A, bool**&B, long long k, long long k_max, long long k_diameter) {
    long long k_last = 0;
    long long dim = 1;
    while(1) {
        cout << dim << '\n';
        ++dim;
        sparseMultiplication(B, A);
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i != j && B[i][j] && shortest_distance[i][j] == 0) {
                    shortest_distance[i][j] = dim;
                    ++k;
                    if (k > k_max - 1) return ;
                }
            }
        }
        if (k > k_max - 1) return ;
        if (k_diameter == dim) return ;
        if (k == k_last) return ;
        k_last = k;
    }
}

void unweightedPipeline(int n, int nodes, long long**& shortest_distance, bool**& A, bool**& B) {
    // The number of pairs of nodes that have found the shortest path
    long long k = 0; 
    // Maximum number of node pairs
    long long k_max = 1.0 * nodes * (nodes - 1); 
    // Graph diameter
    long long k_diameter = n; 
    // cout << "Please enter the graph diameter: " << '\n';
    // cin >> k_diameter;
    cout << "[default: graph diameter = nodes number]\n";

    Timer bTimer("DAWN");
    bTimer.begin();

    DAWN(shortest_distance, A, B, k, k_max, k_diameter);

    bTimer.end();
}

int main(int argc, char *argv[]) {
    // get file path
    input_path = argv[1];
    out_path = argv[2];
    
    cout << "Please enter the number of nodes in the graph: \n";
    cin >> N;
    int nodes = N;

    // odd matrix
    if (N % 2 != 0) N = N + 1;

    long long** shortest_distance = new long long* [N];
    // const int m = static_cast<const int>(N);
    // bitset<m> A;
    bool** A = new bool* [N];
    bool** B = new bool* [N];
    for (int i = 0; i < N; ++i) {
        shortest_distance[i] = new long long[N]();
        A[i] = new bool[N]();
        B[i] = new bool[N]();
    }

    int select_weight = 0, select_direction = 0;
    cout << "[default: unweighted graph]\n";
    // cout << "Please select the type of diagram: 1-Unweighted graph  2-Weighted graph\n";
    // cin >> select_weight;
    cout << "Please select the type of diagram: 1-Undirected graph  2-Directed graph\n";
    cin >> select_direction;

    writeList(shortest_distance, A, B, select_direction);
    
    unweightedPipeline(N, nodes, shortest_distance, A, B);

    // Output the shortest path result
    printRes(shortest_distance); 

    return 0;
}