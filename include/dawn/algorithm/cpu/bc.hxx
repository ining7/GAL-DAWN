#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>

namespace DAWN {
namespace BC_CPU {
float Betweenness_Centrality(Graph::Graph_t& graph,
                             int source,
                             std::string& output_path);

float Betweenness_Centrality_Weighted(Graph::Graph_t& graph,
                                      int source,
                                      std::string& output_path);
}  // namespace BC_CPU
}  // namespace DAWN