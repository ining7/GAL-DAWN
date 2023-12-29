#include "dawn.hxx"
namespace DAWN {
void Graph::createGraph(std::string& input_path, DAWN::Graph& graph)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }

  std::string line;
  int         rows, cols, nnz;

  std::getline(file, line);
  std::stringstream ss(line);
  std::string       format;
  ss >> format;
  if (format == "%%MatrixMarket") {
    std::string object, format, field, symmetry;
    ss >> object >> format >> field >> symmetry;
    if (symmetry != "symmetric") {
      graph.directed = true;
      std::cout << "directed graph" << std::endl;
    } else {
      graph.directed = false;
      std::cout << "undirected graph" << std::endl;
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
  std::cout << rows << " " << cols << " " << nnz << std::endl;
  file.close();

  graph.rows  = rows;
  graph.cols  = cols;
  graph.entry = 0;

  std::cout << "Read Input Graph" << std::endl;

  if (graph.directed) {
    if (graph.weighted) {
      graph.coo.val = new float[nnz * 2];
      std::fill_n(graph.coo.val, 0.0f, nnz * 2);
      std::cout << "readGraphDW" << std::endl;
      graph.readGraphDW(input_path, graph);

    } else {
      std::cout << "readGraphD" << std::endl;
      graph.readGraphD(input_path, graph);
    }
  } else {
    graph.coo.col = new int[nnz];
    graph.coo.row = new int[nnz];
    std::fill_n(graph.coo.col, 0, nnz);
    std::fill_n(graph.coo.row, 0, nnz);
    if (graph.weighted) {
      graph.coo.val = new float[nnz];
      std::fill_n(graph.coo.val, 0.0f, nnz);
      std::cout << "readGraphW" << std::endl;
      graph.readGraphW(input_path, graph);
    } else {
      std::cout << "readGraph" << std::endl;
      graph.readGraph(input_path, graph);
    }
  }
  std::cout << "Initialize Input Matrices" << std::endl;
}

// 定义一个自定义的比较函数，用于按照 pair
// 的第一个元素升序排列，当第一个元素相同时，按照第二个元素升序排列
auto compare = [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
  if (a.first != b.first) {
    return a.first < b.first;  // 按照 pair 中的第一个元素升序排序
  } else {
    return a.second < b.second;  // 当第一个元素相同时，按照第二个元素升序排序
  }
};

auto compareD = [](const std::tuple<int, int, int>& a,
                   const std::tuple<int, int, int>& b) {
  // 按照第一个元素升序排序
  if (std::get<0>(a) != std::get<0>(b)) {
    return std::get<0>(a) >
           std::get<0>(b);  // 这里是大顶堆，如果需要小顶堆改为 <
  }
  // 如果第一个元素相等，则按照第二个元素升序排序
  if (std::get<1>(a) != std::get<1>(b)) {
    return std::get<1>(a) >
           std::get<1>(b);  // 这里是大顶堆，如果需要小顶堆改为 <
  }
  // 如果前两个元素也相等，则按照第三个元素升序排序
  return std::get<2>(a) > std::get<2>(b);  // 这里是大顶堆，如果需要小顶堆改为 <
};

void Graph::readGraph(std::string& input_path, DAWN::Graph& graph)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  // 使用自定义的比较函数定义优先队列
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                      decltype(compare)>
    s(compare);

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
      s.push({rows, cols});
      s.push({cols, rows});
    }
  }
  file.close();

  graph.nnz     = s.size();
  graph.coo.col = new int[graph.nnz];
  graph.coo.row = new int[graph.nnz];
  std::fill_n(graph.coo.col, graph.nnz, 0);
  std::fill_n(graph.coo.row, graph.nnz, 0);

  while (!s.empty()) {
    graph.coo.row[i] = s.top().first;
    graph.coo.col[i] = s.top().second;
    i++;
    s.pop();
  }

  std::cout << "nnz: " << graph.nnz << std::endl;

  DAWN::Tool tool;
  tool.transpose(graph.nnz, graph.coo);
  tool.coo2Csr(graph.rows, graph.nnz, graph.csrB, graph.coo);

  delete[] graph.coo.col;
  graph.coo.col = NULL;
  delete[] graph.coo.row;
  graph.coo.row = NULL;
}

void Graph::readGraphD(std::string& input_path, DAWN::Graph& graph)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

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
    }
  }
  file.close();

  graph.nnz = i;
  std::cout << "nnz: " << graph.nnz << std::endl;
}

void Graph::readGraphW(std::string& input_path, DAWN::Graph& graph)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  int   rows, cols;
  float vals;
  int   i = 0;

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
      i++;
      graph.coo.row[i] = cols;
      graph.coo.col[i] = rows;
      graph.coo.val[i] = vals;
      i++;
    }
  }
  file.close();

  graph.nnz = i;
  std::cout << "nnz: " << graph.nnz << std::endl;
}

void Graph::readGraphDW(std::string& input_path, DAWN::Graph& graph)
{
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }
  std::string line;

  int   rows, cols;
  float vals;
  int   i = 0;

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
      i++;
    }
  }
  file.close();

  graph.nnz = i;
  std::cout << "nnz: " << graph.nnz << std::endl;
}

void Graph::readList(std::string& input_path, DAWN::Graph& graph)
{
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