#include "dawn.hxx"
namespace DAWN {

void Tool::transport(int n, int nnz, DAWN::Graph::Coo& coo)
{
  std::vector<std::pair<int, int>> tmp;
  for (int i = 0; i < nnz; i++) {
    tmp.push_back({coo.row[i], coo.col[i]});
  }
  std::sort(tmp.begin(), tmp.end());
  for (int i = 0; i < nnz; i++) {
    coo.row[i] = tmp[i].second;
    coo.col[i] = tmp[i].first;
  }
}

void Tool::coo2Csr(int n, int nnz, DAWN::Graph::Csr& csr, DAWN::Graph::Coo& coo)
{
  csr.val     = new float[nnz];
  csr.row_ptr = new int[n + 1];
  csr.col     = new int[nnz];
  // 统计每一列的非零元素数目
  int* row_count = new int[n]();
  for (int i = 0; i < nnz; i++) {
    row_count[coo.col[i]]++;
  }
  csr.row_ptr[0] = 0;
  for (int i = 1; i <= n; i++) {
    csr.row_ptr[i] = csr.row_ptr[i - 1] + row_count[i - 1];
  }
// 将每个非零元素填充到csrval和csrcol中
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
      csr.col[j] = coo.row[j];
      csr.val[j] = coo.val[j];
    }
  }
  delete[] row_count;
}

void Tool::csr2Csm(int n, int nnz, DAWN::Graph::Csm& csm, DAWN::Graph::Csr& csr)
{
  csm.val = new float*[n];
  csm.row = new int[n];
  csm.col = new int*[n];
  std::fill_n(csm.row, n, 0);
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    csm.row[i] = csr.row_ptr[i + 1] - csr.row_ptr[i];
  }

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    if (csm.row[i] == 0) {
      csm.val[i]    = new float[1];
      csm.val[i][0] = 0.0f;
      csm.col[i]    = new int[1];
      csm.col[i][0] = 0;
    }
    csm.val[i] = new float[csm.row[i]];
    csm.col[i] = new int[csm.row[i]];
    std::fill_n(csm.val[i], csm.row[i], 0.0f);
    std::fill_n(csm.col[i], csm.row[i], 0);
  }

// 将每个非零元素填充到csrval和csrcol中
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < csm.row[i]; j++) {
      csm.col[i][j] = csr.col[csr.row_ptr[i] + j];
      csm.val[i][j] = csr.val[csr.row_ptr[i] + j];
    }
  }
}

void Tool::coo2Csm(int n, int nnz, DAWN::Graph::Csm& csm, DAWN::Graph::Coo& coo)
{
  csm.val = new float*[n];
  csm.row = new int[n];
  csm.col = new int*[n];
  std::fill_n(csm.row, n, 0);
  for (int i = 0; i < nnz; i++) {
    csm.row[coo.col[i]]++;
  }
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    if (csm.row[i] == 0) {
      csm.val[i]    = new float[1];
      csm.val[i][0] = 0.0f;
      csm.col[i]    = new int[1];
      csm.col[i][0] = 0;
    }
    csm.val[i] = new float[csm.row[i]];
    csm.col[i] = new int[csm.row[i]];
    std::fill_n(csm.val[i], csm.row[i], 0.0f);
    std::fill_n(csm.col[i], csm.row[i], 0);
  }

  int tmp = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < csm.row[i]; j++) {
      csm.col[i][j] = coo.row[tmp];
      csm.val[i][j] = coo.val[tmp];
      tmp++;
    }
  }
}

void Tool::outfile(int n, int* result, int source, std::string& output_path)
{
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

void Tool::outfile(int                      n,
                   thrust::host_vector<int> result,
                   int                      source,
                   std::string&             output_path)
{
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

void Tool::infoprint(int   entry,
                     int   total,
                     int   interval,
                     int   thread,
                     float elapsed_time)
{
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