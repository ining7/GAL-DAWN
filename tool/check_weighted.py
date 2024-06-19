import networkx as nx
import numpy as np
import argparse
import scipy
import matplotlib.pyplot as plt


# 读取mtx格式的矩阵文件，返回为稀疏矩阵的coo表示格式


def read_mtx(mtx_file_path):
    sparse_matrix = scipy.io.mmread(mtx_file_path)
    G = nx.from_scipy_sparse_array(sparse_matrix, create_using=nx.DiGraph)
    return G

# 计算指定源节点的单源最短路径长度


def sssp(graph, source_node):
    shortest_paths = nx.single_source_dijkstra_path_length(
        graph, source=source_node)
    return shortest_paths


def bf(graph, source_node):
    shortest_paths = nx.single_source_bellman_ford_path_length(
        graph,  source=source_node)
    return shortest_paths


# 创建解析器对象并定义命令行参数
parser = argparse.ArgumentParser(
    description="Calculate shortest path length from source node in mtx graph file")
parser.add_argument("algorithm", metavar="ALGORITHM", type=str,
                    help="Algorithm Dijkstra and Bellman-Ford")
parser.add_argument("input_file", metavar="INPUT_FILE",
                    type=str, help="path to input mtx graph file")
parser.add_argument("output_file", metavar="OUTPUT_FILE",
                    type=str, help="path to output file")
parser.add_argument("source_node", metavar="SOURCE_NODE",
                    type=int, help="source node index (0-based)")

# 解析命令行参数
args = parser.parse_args()


G = read_mtx(args.input_file)
# 删除从节点到自身的环
self_loops = list(nx.nodes_with_selfloops(G))
G.remove_edges_from([(node, node) for node in self_loops])

if args.algorithm == 'sssp':
    shortest_paths = sssp(G, args.source_node)
if args.algorithm == 'bf':
    shortest_paths = bf(G, args.source_node)

# 对所有节点按节点编号进行排序
sorted_nodes = sorted(shortest_paths.keys())
with open(args.output_file, "w") as f:
    for node in sorted_nodes:
        length = shortest_paths[node]
        if args.source_node != node:
            f.write("{} {} {:.6f}\n".format(args.source_node, node, length))
