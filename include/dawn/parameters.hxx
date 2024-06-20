// dawn/parameters.hxx
#pragma once
#include <dawn/include.hxx>
#include <getopt.h>

namespace DAWN {
namespace IO {
struct parameters_t {
  std::string input_path;
  std::string output_path;
  std::string sourceList_path;
  bool print = false;
  bool weighted = false;
  int source = -1;
};
parameters_t parameters(int argc, char* argv[]);
void printHelp();
}  // namespace IO
}  // namespace DAWN