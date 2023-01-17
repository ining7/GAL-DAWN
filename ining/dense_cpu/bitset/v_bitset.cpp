#include "access.h"
#include "timer.hpp"
#include "omp.h"

using namespace std;

const int N = 1000 + 10;

string input_path = "";
string out_path = "";

void printRes(int8_t**& shortest_distance);
void writeList(int8_t**& shortest_distance, vector<bitset<N>>& A, vector<bitset<N>>& B, int select_direction, long long& k);
void denseMultiplication(int n, vector<bitset<N>>& A, vector<bitset<N>>& B);
void DAWN(int n, int8_t**& shortest_distance, vector<bitset<N>>& A, vector<bitset<N>>& B, long long k, long long k_max, long long k_diameter);
void unweightedPipeline(int n, int nodes, int8_t**& shortest_distance, vector<bitset<N>>& A, vector<bitset<N>>& B, long long k);

void printRes(int n, int8_t**& shortest_distance) {
    ofstream out(out_path);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << setw(3) << static_cast<int>(shortest_distance[i][j]) << (j == n - 1 ? '\n' : ' ');
        }
    }
    out.close();
}

void writeList(int8_t**& shortest_distance, vector<bitset<N>>& A, vector<bitset<N>>& B, int select_direction, long long& k) {
    ifstream infile;
    infile.open(input_path);
    if(!infile) cout << " === File error ===\n";
    while(infile) {
        if (infile.eof()) break;
        int a, b; 
        infile >> a >> b;
        // transpose A
        if (A[b].test(a) == false) ++k;
        A[b].set(a);
        B[a].set(b);
        shortest_distance[a][b] = 1;
        // if (select_direction == 1) {
        //     A[a].set(b);
        //     B[b].set(a);
        //     shortest_distance[b][a] = 1;
        // }
    }
    infile.close();
}

void denseMultiplication(int n, vector<bitset<N>>& A, vector<bitset<N>>& B) {
    vector<bitset<N>> res(N);
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bool tmp = (A[i] & B[j]).any();
            if (res[i][j] == false && tmp == true) {
                res[i].set(j);
            }
        }
    }
    A = res;
}

void DAWN(int n, int8_t**& shortest_distance, vector<bitset<N>>& A, vector<bitset<N>>& B, long long k, long long k_max, long long k_diameter) {
    long long k_last = 0;
    long long dim = 1;
    while(1) {
        ++dim;
        cout << dim << '\n';
        // Timer bTimer("wh");
        // bTimer.begin();
        denseMultiplication(n, B, A);
        int* cnt = new int[n];
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            cnt[i] = 0;
            for (int j = 0; j < n; ++j) {
                if ((i != j) && B[i].test(j) && (shortest_distance[i][j] == 0)) {
                    shortest_distance[i][j] = (int8_t)dim;
                    ++cnt[i];
                    // if (k > k_max - 1) return ;
                }
            }
        }
        for (int i = 0; i < n; ++i) k += cnt[i];
        // bTimer.end();
        if (k > k_max - 1) return ;
        if (k_diameter == dim) return ;
        if (k == k_last) return ;
        k_last = k;
    }
}

void unweightedPipeline(int n, int nodes, int8_t**& shortest_distance, vector<bitset<N>>& A, vector<bitset<N>>& B, long long k) {
    // Maximum number of node pairs
    long long k_max = 1.0 * nodes * (nodes - 1); 
    // Graph diameter
    long long k_diameter = n; 
    // cout << "Please enter the graph diameter: " << '\n';
    // cin >> k_diameter;
    cout << "[default: graph diameter = nodes number]\n";

    Timer bTimer("DAWN");
    bTimer.begin();

    DAWN(n, shortest_distance, A, B, k, k_max, k_diameter);

    bTimer.end();
}

int main(int argc, char *argv[]) {
    // get file path
    input_path = argv[1];
    // out_path = argv[2];

    cout << "Please enter the number of nodes in the graph: \n";
    int n; cin >> n;
    int nodes = n;

    // odd matrix
    if (n % 2 != 0) n = n + 1;

    vector<bitset<N>> A(N);
    vector<bitset<N>> B(N);
    int8_t** shortest_distance = new int8_t* [n];
    for (int i = 0; i < n; ++i) {
        shortest_distance[i] = new int8_t[n]();
    }

    int select_weight = 0, select_direction = 0;
    cout << "[default: unweighted graph]\n";
    // cout << "Please select the type of diagram: 1-Unweighted graph  2-Weighted graph\n";
    // cin >> select_weight;
    cout << "Please select the type of diagram: 1-Undirected graph  2-Directed graph\n";
    cin >> select_direction;

    long long k = 0;
    writeList(shortest_distance, A, B, select_direction, k);
    
    unweightedPipeline(n, nodes, shortest_distance, A, B, k);

    // Output the shortest path result
    // printRes(n, shortest_distance); 

    return 0;
}