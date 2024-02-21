#include <dawn/algorithm/gpu/sssp.cuh>
#include <dawn/algorithm/gpu/bfs.cuh>

namespace DAWN {
namespace APSP_GPU {

float run(Graph::Graph_t& graph, std::string& output_path);

float run_Weighted(Graph::Graph_t& graph, std::string& output_path);
}  // namespace APSP_GPU
}  // namespace DAWN