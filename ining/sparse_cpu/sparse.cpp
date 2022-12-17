#include "access.h"

using namespace std;

map<pair<int, int>, int> tuple_to_pair(vector<tuple<int, int, int>> A);
vector<tuple<int, int, int>> pair_to_tuple(map<pair<int, int>, int> A);
void print_pair(map<pair<int, int>, int> A);
void print_tuple(vector<tuple<int, int, int>> A);
vector<tuple<int, int, int>> sparse_multiplication(int n, vector<tuple<int, int, int>> a, int m, vector<tuple<int, int, int>> b);
void unweighted_Coppersmith_power(vector<tuple<int, int, int>> &len, int n, int max, int select2);
int unweighted_assignment(vector<tuple<int, int, int>> &A, vector<tuple<int, int, int>> &B, vector<tuple<int, int, int>> &len, long long k, int max, int select);

// change vector<tuple<int, int, int>> to map<pair<int, int>, int>
map<pair<int, int>, int> tuple_to_pair(vector<tuple<int, int, int>> A) {
    map<pair<int, int>, int> res;
    for (auto it : A) {
        res[make_pair(get<0>(it), get<1>(it))] = get<2>(it);
    }
    return res;
}

// change map<pair<int, int>, int> to vector<tuple<int, int, int>>
vector<tuple<int, int, int>> pair_to_tuple(map<pair<int, int>, int> A) {
    vector<tuple<int, int, int>> res;
    for (auto it : A) {
        if (it.second) {
            res.emplace_back(make_tuple(it.first.first, it.first.second, it.second));
        }
    }
    return res;
}

// Output the pair type as an adjacency matrix
void print_pair(int n, map<pair<int, int>, int> A) {
    cout << "\n ===== output adjacency matrix:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A[make_pair(i, j)]) cout << setw(7) << A[make_pair(i, j)] << ' ';
            else cout << setw(7) << 0 << ' ';
        }
        cout << '\n';
    }
    cout << " === end of adjacency matrix output\n\n";
}

// Output the tuple type as an adjacency matrix
void print_tuple(int n, vector<tuple<int, int, int>> A) {
    cout << "\n ===== output adjacency matrix:\n";
    map<pair<int, int>, int> tmp;
    for (auto it : A) {
        n = max({n, get<0>(it), get<1>(it)});
        tmp[make_pair(get<0>(it), get<1>(it))] = get<2>(it);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (tmp[make_pair(i, j)]) {
                cout << setw(7) << tmp[make_pair(i, j)] << ' ';
            } else cout << setw(7) << 0 << ' ';
        }
        cout << '\n';
    }
    cout << " === end of adjacency matrix output\n\n";
}

// sparse matrix multiplication
vector<tuple<int, int, int>> sparse_multiplication(int n, vector<tuple<int, int, int>> a, int m, vector<tuple<int, int, int>> b) {
    vector<tuple<int, int, int>> c;
    map<pair<int, int>, int> tmp;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (get<1>(a[i]) == get<0>(b[j])) {
                tmp[make_pair(get<0>(a[i]), get<1>(b[j]))] += get<2>(a[i]) * get<2>(b[j]);
            }
        }
    }
    for (auto it : tmp) {
        if (it.second) {
            c.emplace_back(make_tuple(it.first.first, it.first.second, it.second));
        }
    }
    return c;
}

// read in unweighted graph from dataset
int unweighted_assignment(vector<tuple<int, int, int>> &A, vector<tuple<int, int, int>> &B, vector<tuple<int, int, int>> &len, long long k, int max, int select) {
    //输入邻接矩阵

    //文件形式读入-数据集
//    cout << "请输入待测试文件名称，并保证该文件与此文件在同一目录下" << '\n';
//    string path;
//    cin >> path;
//    cout << path << '\n';
//修改读入流

    //输入
    cout << "input A: \n";
    for (int i = 0; i < max; ++i) {
        for (int j = 0; j < max; ++j) {
            int tmp = 0;
            cin >> tmp;
            if (tmp == 1) {
                A.emplace_back(make_tuple(i, j, 1));
                B.emplace_back(make_tuple(i, j, 1));
                len.emplace_back(make_tuple(i, j, 1));
                if (select == 1) {
                    A.emplace_back(make_tuple(j, i, 1));
                    B.emplace_back(make_tuple(j, i, 1));
                    len.emplace_back(make_tuple(j, i, 1));
                }
            } 
        }
    }
    return k;
}

// DAWN - Unweighted graph
void unweighted_Coppersmith_power(vector<tuple<int, int, int>> &len, int n, int max, int select) {
    //初始化矩阵A与矩阵B
    vector<tuple<int, int, int>> A;
    vector<tuple<int, int, int>> B;

    long long k = 0; //记录已经找到的最短路径的节点对数
    long long k_max = 1.0 * max * (max - 1); //节点对数量上限
    long long k_last = 0; //上一次循环的k值

    //输入数据
    cout << "输入网络直径" << '\n';
    int k_diameter = 0;
    cin >> k_diameter;
    k = unweighted_assignment(A, B, len, k, max, select);
    cout << "数据初始化完成" << '\n';

    //开始计时
    clock_t start, finish;
    start = clock();
    cout << "计算计时开始" << endl;
    int dim = 1;
    while (1) {
        ++dim;
        B = sparse_multiplication(B.size(), B, A.size(), A);

        map<pair<int, int>, int> a = tuple_to_pair(A);
        map<pair<int, int>, int> b = tuple_to_pair(B);
        map<pair<int, int>, int> l = tuple_to_pair(len);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (b[make_pair(i, j)] != 0 && l[make_pair(i, j)] == 0 && i != j) {
                    l[make_pair(i, j)] = dim;
                    ++k;
                    if (k > k_max - 1) break;
                }
            }
            if (k > k_max - 1) break;
        }

        len.clear();
        len = pair_to_tuple(l);

        if (k > k_max - 1) break;
        if (k_diameter == dim) break;
        k_last = k;
    }
    finish = clock();
    cout << "程序运行总时间为" << double(finish - start) / CLOCKS_PER_SEC << "s\n";
}

int main() {
    int n = 0;
    cout << "请输入矩阵的阶数：\n";
    cin >> n;
    int max = n;
    // odd matrix
    if (n % 2 != 0) n = n + 1;
    vector<tuple<int, int, int>> len;
    int select1 = 0, select2 = 0;
    cout << "当前默认为无权图\n";
//    cout << "请选择图的类型：1-无权  2-有权\n";
//    cin >> select1;
    cout << "请选择图的类型：1-无向  2-有向\n";
    cin >> select2;
    unweighted_Coppersmith_power(len, n, max, select2);
    print_tuple(n, len);
    return 0;
}