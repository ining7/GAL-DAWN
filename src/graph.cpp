/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/dawn.hxx>
namespace DAWN {

void Graph::transpose_Weighted(int nnz, DAWN::Graph::Coo_t& coo) {
  std::vector<std::pair<int, std::pair<int, float>>> tmp;
  for (int i = 0; i < nnz; i++) {
    tmp.push_back({coo.row[i], {coo.col[i], coo.val[i]}});
  }
  std::sort(tmp.begin(), tmp.end());
  for (int i = 0; i < nnz; i++) {
    coo.row[i] = tmp[i].second.first;
    coo.col[i] = tmp[i].first;
    coo.val[i] = tmp[i].second.second;
  }
}

void Graph::transpose(int nnz, DAWN::Graph::Coo_t& coo) {
  std::vector<std::pair<int, int>> tmp;
  for (int i = 0; i < nnz; i++) {
    tmp.push_back({coo.row[i], coo.col[i]});
  }
  std::sort(tmp.begin(), tmp.end());
  for (int i = 0; i < nnz; i++) {
    coo.row[i] = tmp[i].first;
    coo.col[i] = tmp[i].second;
  }
}

void Graph::coo2Csr_Weighted(int n,
                             int nnz,
                             DAWN::Graph::Csr_t& csr,
                             DAWN::Graph::Coo_t& coo) {
  csr.val = new float[nnz];
  csr.row_ptr = new int[n + 1];
  csr.col = new int[nnz];

  // Count the number of non-zero elements in each column
  int* row_count = new int[n]();
  for (int i = 0; i < nnz; i++) {
    row_count[coo.row[i]]++;
  }
  csr.row_ptr[0] = 0;
  for (int i = 1; i <= n; i++) {
    csr.row_ptr[i] = csr.row_ptr[i - 1] + row_count[i - 1];
  }

// Fill each non-zero element into val and col
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
      csr.col[j] = coo.col[j];
      csr.val[j] = coo.val[j];
    }
  }
  delete[] row_count;
}

void Graph::coo2Csr(int n,
                    int nnz,
                    DAWN::Graph::Csr_t& csr,
                    DAWN::Graph::Coo_t& coo) {
  csr.row_ptr = new int[n + 1];
  csr.col = new int[nnz];

  // Count the number of non-zero elements in each column
  int* row_count = new int[n]();
  for (int i = 0; i < nnz; i++) {
    row_count[coo.row[i]]++;
  }
  csr.row_ptr[0] = 0;
  for (int i = 1; i <= n; i++) {
    csr.row_ptr[i] = csr.row_ptr[i - 1] + row_count[i - 1];
  }

// Fill each non-zero element into val and col
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
      csr.col[j] = coo.col[j];
    }
  }
  delete[] row_count;
}

void Graph::createGraph(std::string& input_path, Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }

  std::string line;
  int rows, cols, nnz;

  std::getline(file, line);
  std::stringstream ss(line);
  std::string format;
  ss >> format;
  if (format == "%%MatrixMarket") {
    std::string object, format, field, symmetry;
    ss >> object >> format >> field >> symmetry;
    if (symmetry != "symmetric") {
      graph.directed = true;
      std::cout << "Genral" << std::endl;
    } else {
      graph.directed = false;
    }
  } else {
    std::cout << "invalid file" << std::endl;
    return;
  }

  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols >> nnz;
    break;
  }
  file.close();

  graph.rows = rows;
  graph.cols = cols;
  graph.nnz = nnz;
  graph.entry = 0;

  std::cout << "Read Input Graph" << std::endl;

  if (graph.directed) {
    if (graph.weighted)
      readGraph_Directed_Weighted(input_path, graph);
    else
      readGraph_Directed(input_path, graph);
  } else {
    if (graph.weighted)
      readGraph_Weighted(input_path, graph);
    else
      readGraph(input_path, graph);
  }
}

void Graph::readGraph(std::string& input_path, Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;
  std::priority_queue<std::pair<int, int>> s;
  int rows, cols;

  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols;
    rows--;
    cols--;
    if (rows != cols) {
      s.push({rows, cols});
      s.push({cols, rows});
    }
  }
  file.close();

  graph.nnz = s.size();
  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);
  int i = graph.nnz - 1;

  while (!s.empty()) {
    graph.coo.row[i] = s.top().first;
    graph.coo.col[i] = s.top().second;
    --i;
    s.pop();
  }
  coo2Csr(graph.rows, graph.nnz, graph.csr, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
}

void Graph::readGraph_Directed(std::string& input_path, Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);

  int rows, cols;
  int i = 0;

  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols;
    rows--;
    cols--;
    if (rows != cols) {
      graph.coo.row[i] = rows;
      graph.coo.col[i] = cols;
      ++i;
    }
  }
  file.close();

  graph.nnz = i;
  std::cout << "nnz: " << graph.nnz << std::endl;

  transpose(graph.nnz, graph.coo);
  coo2Csr(graph.rows, graph.nnz, graph.csr, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
}

void Graph::readGraph_Weighted(std::string& input_path, Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  std::priority_queue<std::tuple<int, int, float>> s;

  int rows, cols;
  float vals;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols >> vals;
    rows--;
    cols--;
    if (rows != cols) {
      s.push({rows, cols, vals});
      s.push({cols, rows, vals});
    }
  }
  file.close();

  graph.nnz = s.size();
  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  graph.coo.val = new float[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);
  std::fill_n(graph.coo.val, graph.nnz, 0.0f);
  int i = graph.nnz - 1;

  while (!s.empty()) {
    graph.coo.row[i] = std::get<0>(s.top());
    graph.coo.col[i] = std::get<1>(s.top());
    graph.coo.val[i] = std::get<2>(s.top());
    --i;
    s.pop();
  }

  coo2Csr_Weighted(graph.rows, graph.nnz, graph.csr, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
  delete[] graph.coo.val;
  graph.coo.val = NULL;
}

void Graph::readGraph_Directed_Weighted(std::string& input_path,
                                        Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  graph.coo.val = new float[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);
  std::fill_n(graph.coo.val, graph.nnz, 0.0f);

  int rows, cols;
  float vals;
  int i = 0;
  std::cout << "nnz: " << graph.nnz << std::endl;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols >> vals;
    rows--;
    cols--;
    if (rows != cols) {
      graph.coo.row[i] = rows;
      graph.coo.col[i] = cols;
      graph.coo.val[i] = vals;
      ++i;
    }
  }
  file.close();

  graph.nnz = i;
  std::cout << "nnz: " << graph.nnz << std::endl;

  transpose_Weighted(graph.nnz, graph.coo);
  coo2Csr_Weighted(graph.rows, graph.nnz, graph.csr, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
  delete[] graph.coo.val;
  graph.coo.val = NULL;
}

void Graph::readList(std::string& input_path, Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  int source;
  int i = 0;
  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> source;
    graph.msource.push_back(source);
    i++;
  }
  file.close();
}

}  // namespace DAWN