#include <dawn/dawn.hxx>

namespace DAWN {
namespace SSSP_CPU {

// Shortest Path Algorithm
float runSSSP(Graph::Graph_t& graph, std::string& output_path);

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