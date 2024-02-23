/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>
namespace DAWN {
namespace APSP_CPU {

// Shortest Path Algorithm
float runTG(Graph::Graph_t& graph, std::string& output_path);

float runSG(Graph::Graph_t& graph, std::string& output_path);

}  // namespace APSP_CPU
}  // namespace DAWN