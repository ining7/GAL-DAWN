/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/dawn.hxx>

namespace DAWN {
namespace SSSP_CPU {

// Shortest Path Algorithm
float run(Graph::Graph_t& graph, std::string& output_path);

float SSSP(int* row_ptr,
           int* col,
           float* val,
           int row,
           int source,
           bool print,
           std::string& output_path);

float SSSP_kernel(int* row_ptr,
                  int* col,
                  float* val,
                  int row,
                  int source,
                  bool print,
                  std::string& output_path);

int GOVM(int* row_ptr,
         int* col,
         float* val,
         int row,
         int*& alpha,
         int*& beta,
         float*& distance,
         int entry);

bool GOVMP(int* row_ptr,
           int* col,
           float* val,
           int row,
           bool*& alpha,
           bool*& beta,
           float*& distance);

}  // namespace SSSP_CPU
}  // namespace DAWN