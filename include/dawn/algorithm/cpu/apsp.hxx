#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>
namespace DAWN {
namespace APSP_CPU {

// Shortest Path Algorithm
float runAPSPTG(Graph::Graph_t& graph, std::string& output_path);

float runAPSPSG(Graph::Graph_t& graph, std::string& output_path);
}  // namespace APSP_CPU
}  // namespace DAWN