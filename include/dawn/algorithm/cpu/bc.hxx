/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-04-21
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>

namespace DAWN {
namespace BC_CPU {

float Betweenness_Centrality(Graph::Graph_t& graph, std::string& output_path);

float test(DAWN::Graph::Graph_t& graph, std::string& output_path);

float SOVM(int* row_ptr, int* col, int row, int source, float*& bc_temp);

bool kernel(int* row_ptr,
            int* col,
            int row,
            bool*& alpha,
            bool*& beta,
            bool*& gamma,
            int*& amount,
            std::vector<std::queue<int>>& path,
            std::deque<int>& path_length,
            float*& bc_temp,
            int step,
            bool is_converged,
            int source);
void accumulate(std::vector<std::queue<int>>& path,
                std::deque<int>& path_length,
                float*& bc_temp,
                int source);
}  // namespace BC_CPU
}  // namespace DAWN