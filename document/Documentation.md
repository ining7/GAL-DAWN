# 1 Function
Functions in DAWN are categorized into user-facing API functions and kernel functions. Here, we will only explain the more comprehensible and user-friendly API functions.

We categorize API functions into different levels: DAWN and direct application functions are classified as Level-1, algorithms such as betweenness centrality that utilize DAWN for graph computations are classified as Level-2, and algorithms like community detection that use both Level-1 and Level-2 functions to achieve specific metrics are classified as Level-3.

## 2.1 Data structure and I/O Function
In the near future, we will make some modifications to the data structures and I/O functions. Once these modifications are complete, this document will be updated accordingly.

## 3.1 Level-1 API Function
Level-1 API functions are those with a time complexity of less than \(O(n^2)\), and currently includes the following algorithms:
| **BFS**|**SSSP**|
| ------ | ------ | 

### 3.1.1 BFS API
```cpp
DAWN::BFS_CPU::run(Graph::Graph_t& graph, std::string& output_path)
DAWN::BFS_GPU::run(Graph::Graph_t& graph, std::string& output_path)
```
The function is for Breath-First Search via unweighted version of DAWN.
- parameter: A graph of Graph::Graph_t, output file path;
- return: Running time of the task.

### 3.1.2 SSSP API
```cpp
DAWN::SSSP_CPU::run(Graph::Graph_t& graph, std::string& output_path)
DAWN::SSSP_GPU::run(Graph::Graph_t& graph, std::string& output_path)
```
The function is for single source shortest path problem via weighted version of DAWN.
- parameter: A graph of Graph::Graph_t, output file path;
- return: Running time of the task.

## 3.2 Level-2 API Function
Level-2 API functions are those with a time complexity of less than \(O(n^3)\), and currently includes the following algorithms:
| **MSSP**|**APSP**|**Betweenness Centrality**|**Closeness Centrality**|
| ------ | ------ | ------ | ------ | 

### 3.2.1 MSSP API
```cpp
DAWN::MSSP_CPU::run(Graph::Graph_t& graph, std::string& output_path)
DAWN::MSSP_GPU::run(Graph::Graph_t& graph, std::string& output_path)
```
The function is for multi-source shortest path problem via DAWN.
- parameter: A graph of Graph::Graph_t, output file path;
- return: Running time of the task.

### 3.2.2 APSP API
```cpp
DAWN::APSP_CPU::run(Graph::Graph_t& graph, std::string& output_path)
DAWN::APSP_GPU::run(Graph::Graph_t& graph, std::string& output_path)
```
The function is for all-pairs shortest path problem via DAWN.
- parameter: A graph of Graph::Graph_t, output file path;
- return: Running time of the task.

### 3.2.4 Closeness Centrality API
```cpp
DAWN::CC_CPU::run(Graph::Graph_t& graph, std::string& output_path)
DAWN::CC_GPU::run(Graph::Graph_t& graph, std::string& output_path)
```
The function is for Closeness Centrality via DAWN. 
- parameter: A graph of Graph::Graph_t, output file path;
- return: Running time of the task.


### 3.2.4 Betweenness Centrality API
```cpp
DAWN::BC_CPU::run(Graph::Graph_t& graph, std::string& output_path)
```
The function is for Betweenness Centrality via BCDAWN. BCDAWN is a novel algorithm, using DAWN for graph traveling and faster accumulate method with lower-memory requirement.
- parameter: A graph of Graph::Graph_t, output file path;
- return: Running time of the task.

## 3.3 Level-3 API Function
Level-3 API functions are those with a time complexity of more than \(O(n^3)\), and maybe include the following algorithms in the future:
| **Cluster Analysis**|**Community Detection**|
| ------ | ------ |
