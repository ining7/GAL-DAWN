#include <dawn/dawn.hxx>
namespace DAWN {
void Graph::createGraph(std::string& input_path, DAWN::Graph& graph) {
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
    if (graph.weighted) {
      graph.readGraphDW(input_path, graph);

    } else {
      graph.readGraphD(input_path, graph);
    }
  } else {
    if (graph.weighted) {
      graph.readGraphW(input_path, graph);
    } else {
      graph.readGraph(input_path, graph);
    }
  }
}

void Graph::readGraph(std::string& input_path, DAWN::Graph& graph) {
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
  DAWN::Tool tool;
  tool.coo2Csr(graph.rows, graph.nnz, graph.csrB, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
}

void Graph::readGraphD(std::string& input_path, DAWN::Graph& graph) {
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

  DAWN::Tool tool;
  tool.transpose(graph.nnz, graph.coo);
  tool.coo2Csr(graph.rows, graph.nnz, graph.csrB, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
}

void Graph::readGraphW(std::string& input_path, DAWN::Graph& graph) {
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

  DAWN::Tool tool;
  tool.coo2CsrW(graph.rows, graph.nnz, graph.csrB, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
  delete[] graph.coo.val;
  graph.coo.val = NULL;
}

void Graph::readGraphDW(std::string& input_path, DAWN::Graph& graph) {
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

  DAWN::Tool tool;
  tool.transposeW(graph.nnz, graph.coo);
  tool.coo2CsrW(graph.rows, graph.nnz, graph.csrB, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
  delete[] graph.coo.val;
  graph.coo.val = NULL;
}

void Graph::readList(std::string& input_path, DAWN::Graph& graph) {
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