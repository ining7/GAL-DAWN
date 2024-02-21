#include <dawn/dawn.cuh>

namespace DAWN {
namespace BC_GPU {
float Betweenness_Centrality(Graph::Graph_t& graph, std::string& output_path);

float Betweenness_Centrality_Weighted(Graph::Graph_t& graph,
                                      std::string& output_path);
}  // namespace BC_GPU
}  // namespace DAWN