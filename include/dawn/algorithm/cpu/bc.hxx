/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>

namespace DAWN {
namespace BC_CPU {

float kernel(Graph::Graph_t& graph, int source, std::string& output_path);

float kernel_Weighted(Graph::Graph_t& graph,
                      int source,
                      std::string& output_path);

float Betweenness_Centrality(Graph::Graph_t& graph, std::string& output_path);

float Betweenness_Centrality_Weighted(Graph::Graph_t& graph,
                                      std::string& output_path);

int SOVM(Graph::Graph_t& graph,
         int*& alpha,
         int*& beta,
         int*& distance,
         int step,
         int entry);

int GOVM(Graph::Graph_t& graph,
         int*& alpha,
         int*& beta,
         float*& distance,
         int entry);

}  // namespace BC_CPU
}  // namespace DAWN