#include "bfs_cpu.hxx"  // Reference implementation
#include "omp.h"
#include "sssp_cpu.hxx"
#include <chrono>
#include <fstream>
#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/algorithms/sssp.hxx>
#include <iostream>
#include <sstream>
#include <string>

using namespace gunrock;
using namespace memory;

void test_sssp(int num_arguments, char** argument_array) {
  if (num_arguments != 3) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx sourceList.txt"
              << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  csr_t csr;
  std::string filename = argument_array[1];
  std::string sourceList = argument_array[2];

  if (util::is_market(filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
  }

  std::ifstream file(sourceList);
  std::vector<int> sourcelist;
  if (!file.is_open()) {
    std::cerr << "Error opening file " << sourceList << std::endl;
    return;
  } else {
    std::string line;
    int source;
    int i = 0;
    while (std::getline(file, line)) {
      if (line[0] == '%')
        continue;
      std::stringstream ss(line);
      ss >> source;
      sourcelist.push_back(source);
      i++;
    }
    file.close();
  }

  thrust::device_vector<weight_t> nonzero_values(csr.number_of_nonzeros, 1.0f);
  thrust::copy(nonzero_values.begin(), nonzero_values.end(),
               csr.nonzero_values.begin());
  // initialize nonzero_values with 1.0 using
  // device_ptr and std::fill_n

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );  // supports row_indices and column_offsets (default = nullptr)

  // --
  // Params and memory allocation
  srand(time(NULL));

  vertex_t n_vertices = G.get_number_of_vertices();
  vertex_t single_source = 0;  // rand() % n_vertices;
  std::cout << "Single Source = " << single_source << std::endl;

  // --
  // GPU Run

  /// An example of how one can use std::shared_ptr to allocate memory on the
  /// GPU, using a custom deleter that automatically handles deletion of the
  /// memory.
  // std::shared_ptr<weight_t> distances(
  //     allocate<weight_t>(n_vertices * sizeof(weight_t)),
  //     deleter_t<weight_t>());
  // std::shared_ptr<vertex_t> predecessors(
  //     allocate<vertex_t>(n_vertices * sizeof(vertex_t)),
  //     deleter_t<vertex_t>());

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  // --
  // Run problem
  float gpu_elapsed = 0.0f;
  float time_operation = 0.0f;
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  for (auto i = 0; i < sourcelist.size(); i++) {
    single_source = sourcelist[i] % n_vertices;
    auto start = std::chrono::high_resolution_clock::now();

    gpu_elapsed += gunrock::sssp::run(G, single_source, distances.data().get(),
                                      predecessors.data().get());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_operation += elapsed_seconds.count();

    if (i % (sourcelist.size() / 100) == 0) {
      float completion_percentage = static_cast<float>(i * 100.0f) /
                                    static_cast<float>(sourcelist.size());
      std::cout << "Progress: " << completion_percentage << "%" << std::endl;
      std::cout << "Elapsed Time :" << time_operation << " s" << std::endl;
    }
  }
  // Log
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  std::cout << "Operation Time :" << time_operation << std::endl;
  std::cout << "Elapsed Time :" << gpu_elapsed / 1000 << std::endl;
}

void test_bfs(int num_arguments, char** argument_array) {
  if (num_arguments != 3) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx sourceList.txt"
              << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  csr_t csr;
  std::string filename = argument_array[1];
  std::string sourceList = argument_array[2];

  if (util::is_market(filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
  }

  std::ifstream file(sourceList);
  std::vector<int> sourcelist;
  if (!file.is_open()) {
    std::cerr << "Error opening file " << sourceList << std::endl;
    return;
  } else {
    std::string line;
    int source;
    int i = 0;
    while (std::getline(file, line)) {
      if (line[0] == '%')
        continue;
      std::stringstream ss(line);
      ss >> source;
      sourcelist.push_back(source);
      i++;
    }
    file.close();
  }

  thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::device_vector<vertex_t> column_indices(csr.number_of_nonzeros);
  thrust::device_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  // --
  // Build graph + metadata

  auto G =
      graph::build::from_csr<memory_space_t::device,
                             graph::view_t::csr /* | graph::view_t::csc */>(
          csr.number_of_rows,               // rows
          csr.number_of_columns,            // columns
          csr.number_of_nonzeros,           // nonzeros
          csr.row_offsets.data().get(),     // row_offsets
          csr.column_indices.data().get(),  // column_indices
          csr.nonzero_values.data().get(),  // values
          row_indices.data().get(),         // row_indices
          column_offsets.data().get()       // column_offsets
      );

  // --
  // Params and memory allocation
  vertex_t single_source = 0;  // rand() % n_vertices;

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  // --
  // Run problem
  float gpu_elapsed = 0.0f;
  float time_operation = 0.0f;
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  for (auto i = 0; i < sourcelist.size(); i++) {
    single_source = sourcelist[i] % n_vertices;
    auto start = std::chrono::high_resolution_clock::now();

    gpu_elapsed += gunrock::bfs::run(G, single_source, distances.data().get(),
                                     predecessors.data().get());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_operation += elapsed_seconds.count();

    if (i % (sourcelist.size() / 100) == 0) {
      float completion_percentage = static_cast<float>(i * 100.0f) /
                                    static_cast<float>(sourcelist.size());
      std::cout << "Progress: " << completion_percentage << "%" << std::endl;
      std::cout << "Elapsed Time :" << time_operation << " s" << std::endl;
    }
  }
  // --
  // Log
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  std::cout << "Operation Time :" << time_operation << std::endl;
  std::cout << "Elapsed Time :" << gpu_elapsed / 1000 << std::endl;
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
  test_bfs(argc, argv);
}
