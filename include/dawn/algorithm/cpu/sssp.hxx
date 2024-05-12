/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/dawn.hxx>

namespace DAWN {
namespace SSSP_CPU {

// Shortest Path Algorithm
float run(Graph::Graph_t& graph, std::string& output_path);

// kernel
float SSSPs(Graph::Graph_t& graph, int source, std::string& output_path);

float SSSPp(Graph::Graph_t& graph, int source, std::string& output_path);

int GOVM(Graph::Graph_t& graph,
         int*& alpha,
         int*& beta,
         float*& distance,
         int entry);

bool GOVMP(Graph::Graph_t& graph, bool*& alpha, bool*& beta, float*& distance);

}  // namespace SSSP_CPU
}  // namespace DAWN