#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>

namespace DAWN {
namespace MSSP_CPU {

// Shortest Path Algorithm
float runMSSPTG(Graph::Graph_t& graph, std::string& output_path);

float runMSSPSG(Graph::Graph_t& graph, std::string& output_path);

}  // namespace MSSP_CPU
}  // namespace DAWN