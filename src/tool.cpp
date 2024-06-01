#include <dawn/tool.hxx>

float DAWN::Tool::average(int* result, int n) {
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

float DAWN::Tool::average(float* result, int n) {
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

void DAWN::Tool::outfile(int n,
                         int* result,
                         int source,
                         std::string& output_path) {
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

void DAWN::Tool::outfile(int n,
                         float* result,
                         int source,
                         std::string& output_path) {
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

void DAWN::Tool::outfile(int n, int* result, std::string& output_path) {
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if (result[j] > 0)
      outfile << j << " " << result[j] << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

void DAWN::Tool::outfile(int n, float* result, std::string& output_path) {
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file " << output_path << std::endl;
    return;
  }
  std::cout << "Start outfile" << std::endl;
  for (int j = 0; j < n; j++) {
    if (result[j] > 0)
      outfile << j << " " << std::fixed << std::setprecision(6) << result[j]
              << std::endl;
  }
  std::cout << "End outfile" << std::endl;
  outfile.close();
}

void DAWN::Tool::infoprint(int entry,
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
