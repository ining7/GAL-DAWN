/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>

namespace DAWN {
namespace CC_CPU {

// Centrality Algorithm
float run(Graph::Graph_t& graph, int source, float& elapsed_time);

float run_Weighted(Graph::Graph_t& graph, int source, float& elapsed_time);

float kernel(int* row_ptr, int* col, int row, int source, float& elapsed_time);

float kernel_Weighted(int* row_ptr,
                      int* col,
                      float* val,
                      int row,
                      int source,
                      float& elapsed_time);

}  // namespace CC_CPU
}  // namespace DAWN