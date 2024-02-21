#include <dawn/include.hxx>
namespace DAWN {
namespace Tool {
float average(int* result, int n);  // average value of a list

float average(float* result, int n);

void infoprint(int entry,
               int total,
               int interval,
               int thread,
               float elapsed_time);  // Print current task progress

void outfile(int n, int* result, int source, std::string& output_path);

void outfile(int n, float* result, int source, std::string& output_path);
}  // namespace Tool
}  // namespace DAWN