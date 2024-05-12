# How to write a new graph algorithm

This document is intended as a reference for someone to implement a new graph algorithm based on DAWN. As an example, I’ll walk you through implementing a simple algorithm (Breadth-First Search). Let's get started!

## 1. Define the interface for the new algorithm

Create a new file in the `include/dawn/algorithm/cpu` directory called `example.hxx`. You should choose the cpu or gpu implementation based on the target device. We’ll use the cpu implementation as an example.

```cpp
#include <dawn/dawn.hxx>

namespace DAWN {
namespace Example_CPU {

// run
Type run(Graph::Graph_t& graph,......);

// kernel
Type kernel(Graph::Graph_t& graph,......);

}  // namespace Example_CPU
}  // namespace DAWN
```

This includes the function signature, input and output types, and any additional parameters or constraints. We suggest to divide the run and kernel function, which will help you to locate the problems.

## 2. Implement the new algorithm

Create a new file in the `src/dawn/algorithm/cpu` directory called `example.cpp`.

```cpp
#include <dawn/algorithm/cpu/example.hxx>

// If you want to use the component of DAWN, you can include the hxx, such as,
// #include <dawn/algorithm/cpu/bfs.hxx>

Type DAWN::Example_CPU::run(Graph::Graph_t& graph,......)
{
    // ...
    //  float result = DAWN::BFS_CPU::run(graph, output_path);
    return result;
}

Type DAWN::Example_CPU::kernel(Graph::Graph_t& graph,......)
{
    // ...
    return result;
}

```

This includes the function implementation, which should be consistent with the function signature. We do not suggest to use the "using namespace XXX", which may cause errors, such as namespace conflicts or inability to find functions under the namespace. All references and definitions should use full names, including multiple namespaces.

## 3. Add the main function of the new algorithm

Create a new file in the `algorithm/cpu/example/` directory called `example_cpu.cpp`.

```cpp
#include "dawn/algorithm/cpu/example.hxx"

int main(int argc, char** argv)
{ 
    DAWN::Graph::Graph_t graph;
    DAWN::Graph::createGraph(input_path, graph);
    // DAWN::Example_CPU::run(Graph::Graph_t& graph,......)
    return 0;
}

```

This includes the main function of the new algorithm. The input parameters can be read through argv.

## 4. Add the new algorithm to the CMakeLists.txt

The project is structured with a four-tier hierarchy of `CMakeLists.txt` files, designed to facilitate the integration of new components and streamline project management. The incorporation of additional components necessitates minimal modifications, as the majority of the foundational work has been accomplished.

### 4.1. Create `CMakeLists.txt` file for the new algorithm

Copy the `CMakeLists.txt` file from the `algorithm/cpu/XXX/` directory to the `algorithm/cpu/example/` directory. You will get a `CMakeLists.txt` file that looks like this:

```cmake
# Specify source and header files
file(GLOB XXX_SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/algorithm/cpu/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/algorithm/cpu/*.hxx" "${CMAKE_SOURCE_DIR}/include/*.hxx")

# Define the CUDA sources
set(XXX_CPU_SOURCES
    XXX_cpu.cpp
    ${XXX_SOURCES_CPP}
)

# Add the executable
add_executable(xxx_cpu ${XXX_CPU_SOURCES})
target_include_directories(xxx_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Add compile options
target_compile_options(xxx_cpu PUBLIC -O3 -fopenmp)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(xxx_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

```

"CTRL" + "F", use "example" to instead the "xxx". So easy!

### 4.2. Add the new algorithm to the `algorithm/cpu/CMakeLists.txt` file

```cmake
# Add subdirectories conditionally
add_subdirectory(apsp_cpu)
add_subdirectory(mssp_cpu)
add_subdirectory(sssp_cpu)
add_subdirectory(bfs_cpu)
add_subdirectory(cc_cpu)
# Add new algorithm here, like this
add_subdirectory(example_cpu)
```

With these modifications finalized, the remaining two levels of the `cmakelists.txt` will operate autonomously. There is no need for concern if you lack a GPU or are unsure about the GPU architecture; DAWN is designed to autonomously detect the presence of a GPU and identify compatible architectures. However, if you possess knowledge of these specifics, you have the option to specify them within `algorithm/gpu/CMakeLists.txt`.

## 5. Build the project

Now you can compile your whole application as follows:

```bash
mkdir build && cd build
cmake .. && make -j
```

## 6. Run the new algorithm

If compilation succeeds without errors, you can run your code as follows:

```bash
./example_cpu ../data/example.mtx ../data/example_output.txt
```
