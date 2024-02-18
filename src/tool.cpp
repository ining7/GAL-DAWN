#include <dawn/dawn.hxx>
namespace DAWN {

void Tool::transpose_Weighted(int nnz, DAWN::Graph::Coo& coo) {
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

void Tool::transpose(int nnz, DAWN::Graph::Coo& coo) {
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

void Tool::coo2Csr_Weighted(int n,
                    int nnz,
                    DAWN::Graph::Csr& csr,
                    DAWN::Graph::Coo& coo) {
  csr.val = new float[nnz];
  csr.row_ptr = new int[n + 1];
  csr.col = new int[nnz];
  // 统计每一列的非零元素数目
  int* row_count = new int[n]();
  for (int i = 0; i < nnz; i++) {
    row_count[coo.row[i]]++;
  }
  csr.row_ptr[0] = 0;
  for (int i = 1; i <= n; i++) {
    csr.row_ptr[i] = csr.row_ptr[i - 1] + row_count[i - 1];
  }
// 将每个非零元素填充到csrval和csrcol中
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
      csr.col[j] = coo.col[j];
      csr.val[j] = coo.val[j];
    }
  }
  delete[] row_count;
}

void Tool::coo2Csr(int n,
                   int nnz,
                   DAWN::Graph::Csr& csr,
                   DAWN::Graph::Coo& coo) {
  csr.row_ptr = new int[n + 1];
  csr.col = new int[nnz];
  // 统计每一列的非零元素数目
  int* row_count = new int[n]();
  for (int i = 0; i < nnz; i++) {
    row_count[coo.row[i]]++;
  }
  csr.row_ptr[0] = 0;
  for (int i = 1; i <= n; i++) {
    csr.row_ptr[i] = csr.row_ptr[i - 1] + row_count[i - 1];
  }
// 将每个非零元素填充到csr.col中
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
      csr.col[j] = coo.col[j];
    }
  }
  delete[] row_count;
}

float Tool::average(int* result, int n) {
  int64_t sum = 0;
  int i = 0;
  for (int j = 0; j < n; j++) {
    if (result[j] > 0) {
      sum += result[j];
      ++i;
    }
  }
  return 1.0f * sum / i;
}

float Tool::average(float* result, int n) {
  int i = 0;
  float sum = 0.0f;
  float INF = 1.0 * 0xfffffff;
  for (int j = 0; j < n; j++) {
    if (result[j] > 0) {
      sum += result[j];
      ++i;
    }
  }
  return sum / i;
}

void Tool::outfile(int n, int* result, int source, std::string& output_path) {
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if ((source != j) && (result[j] > 0))
      outfile << source << " " << j << " " << result[j] << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

void Tool::outfile(int n, float* result, int source, std::string& output_path) {
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  int INF = 0xfffffff;
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if ((source != j) && (result[j] < INF) && (result[j] > 0))
      outfile << source << " " << j << " " << std::fixed << std::setprecision(6)
              << result[j] << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

void Tool::infoprint(int entry,
                     int total,
                     int interval,
                     int thread,
                     float elapsed_time) {
  if (entry % (total / interval) == 0) {
    float completion_percentage =
        static_cast<float>(entry * 100.0f) / static_cast<float>(total);
    std::cout << "Progress: " << completion_percentage << "%" << std::endl;
    std::cout << "Elapsed Time :"
              << static_cast<double>(elapsed_time) / (thread * 1000) << " s"
              << std::endl;
  }
}

}  // namespace DAWN