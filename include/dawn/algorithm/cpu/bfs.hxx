/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/dawn.hxx>

namespace DAWN {
namespace BFS_CPU {

float run(Graph::Graph_t& graph, std::string& output_path);

// kernel
float BFSp(Graph::Graph_t& graph, int source, std::string& output_path);

float BFSs(Graph::Graph_t& graph, int source, std::string& output_path);

int SOVM(Graph::Graph_t& graph,
         int*& alpha,
         int*& beta,
         int*& distance,
         int step,
         int entry);

bool SOVMP(Graph::Graph_t& graph,
           bool*& alpha,
           bool*& beta,
           int*& distance,
           int step);

}  // namespace BFS_CPU
}  // namespace DAWN