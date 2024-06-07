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


# 创建解析器对象并定义命令行参数
parser = argparse.ArgumentParser(
    description="Calculate shortest path length from source node in mtx graph file")
parser.add_argument("input_file", metavar="INPUT_FILE",
                    type=str, help="path to input mtx graph file")
parser.add_argument("output_file", metavar="OUTPUT_FILE",
                    type=str, help="path to output file")
parser.add_argument("weighted", metavar="WEIGHTED",
                    type=str, help="edge weighted")

# 解析命令行参数
args = parser.parse_args()

G = read_mtx(args.input_file)
# 删除从节点到自身的环
self_loops = list(nx.nodes_with_selfloops(G))
G.remove_edges_from([(node, node) for node in self_loops])

# 打印图的边来验证
# print(G.edges())

# print(nx.number_of_edges(graph) )
print(nx.is_directed(G))

# 介数中心性
bc_value = nx.betweenness_centrality(G, normalized=False, weight=None)
# 对所有节点按节点编号进行排序
sorted_nodes = sorted(bc_value.keys())
with open(args.output_file, "w") as f:
    for node in sorted_nodes:
        value = bc_value[node]
        if value > 0:
            f.write("{} {:.6f}\n".format(node, value))

# # 最短路径
# source_node = 0
# shortest_paths = nx.single_source_shortest_path_length(
#     graph, source=source_node)
# # 对所有节点按节点编号进行排序
# sorted_nodes = sorted(shortest_paths.keys())
# with open(args.output_file, "w") as f:
#     for node in sorted_nodes:
#         length = shortest_paths[node]
#         if source_node != node:
#             f.write("{} {} {}\n".format(source_node, node, length))
# # 使用布局算法来生成规则的平面图
# pos = nx.circular_layout(G)
# plt.figure(figsize=(8, 8))  # 设置图形尺寸为 640*640
# nx.draw(G, pos, with_labels=True, node_color='lightblue',
#         edge_color='grey', arrowsize=20, arrows=False)
# plt.savefig('/home/lxr/code/test/graph.png')  # 保存图形为文件

# # 获取图中所有节点
# nodes = list(G.nodes)

# # 打印所有节点对之间的所有最短路径
# for source in nodes:
#     for target in nodes:
#         if source != target:
#             try:
#                 paths = list(nx.all_shortest_paths(G, source, target))
#                 for path in paths:
#                     print(f"{source} to {target}: {path}")
#             except nx.NetworkXNoPath:
#                 print(f"No path between {source} and {target}")
