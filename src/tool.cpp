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
  for (int j = 0; j < n; j++) {
    if (result[j] > 0) {
      sum += result[j];
      ++i;
    }
  }
  return sum / i;
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
