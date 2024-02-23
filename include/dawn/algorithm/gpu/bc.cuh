/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/dawn.cuh>

namespace DAWN {
namespace BC_GPU {

float Betweenness_Centrality(Graph::Graph_t& graph, std::string& output_path);

float Betweenness_Centrality_Weighted(Graph::Graph_t& graph,
                                      std::string& output_path);

}  // namespace BC_GPU
}  // namespace DAWN