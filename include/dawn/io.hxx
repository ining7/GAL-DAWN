#pragma once
#include <dawn/graph.hxx>

namespace DAWN {
namespace IO {

void readGraph(std::string& input_path, DAWN::Graph::Graph_t& graph);

void readGraph_Weighted(std::string& input_path, DAWN::Graph::Graph_t& graph);

void readGraph_Directed(std::string& input_path, DAWN::Graph::Graph_t& graph);

void readGraph_Directed_Weighted(std::string& input_path,
                                 DAWN::Graph::Graph_t& graph);

void readList(std::string& input_path, DAWN::Graph::Graph_t& graph);

void outfile(int n, int* result, int source, std::string& output_path);

void outfile(int n, float* result, int source, std::string& output_path);

void outfile(int n, int* result, std::string& output_path);

void outfile(int n, float* result, std::string& output_path);
}  // namespace IO
}  // namespace DAWN