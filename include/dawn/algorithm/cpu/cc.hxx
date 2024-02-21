#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>

namespace DAWN {
namespace CC_CPU {

// Centrality Algorithm
float Closeness_Centrality(Graph::Graph_t& graph, int source);

float Closeness_Centrality_Weighted(Graph::Graph_t& graph, int source);
}  // namespace CC_CPU
}  // namespace DAWN