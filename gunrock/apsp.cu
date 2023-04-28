#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <gunrock/algorithms/sssp.hxx>
#include "timer.hpp"
#include <chrono>

using namespace gunrock;
using namespace memory;

void test_apsp(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
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

  if (util::is_market(filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
  }

  // // Set all nonzero values to 1 in parallel
  // #pragma omp parallel for
  // for (int i = 0; i < csr.number_of_nonzeros; i++) {
  //     csr.nonzero_values[i] = 1;
  // }

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
  std::cout << "G.get_number_of_vertices() = " << G.get_number_of_vertices()
            << std::endl;
  std::cout << "G.get_number_of_edges() = " << G.get_number_of_edges()
            << std::endl;

  // --
  // GPU Run

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);
  float gpu_elapsed = 0.0f;
  double time_operation = 0;
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;

  for (auto i = 0; i < n_vertices; i++) {
    single_source = i;

    auto start = std::chrono::high_resolution_clock::now();

    gpu_elapsed += gunrock::sssp::run(G, single_source, distances.data().get(),
                                      predecessors.data().get());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_operation += elapsed_seconds.count();

    if (i % (n_vertices / 1000) == 0) {
      float completion_percentage =
          static_cast<float>(i * 100.0f) / static_cast<float>(n_vertices);
      std::cout << "Progress: " << completion_percentage << "%" << std::endl;
      std::cout << "GPU Elapsed Time :" << time_operation << " s" << std::endl;
    }
  }
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  std::cout << "GPU Elapsed Time :" << time_operation << " s" << std::endl;
}

int main(int argc, char** argv) {
  test_apsp(argc, argv);
  return 0;
}