import networkx as nx
import numpy as np
import argparse

# 读取mtx格式的矩阵文件，返回为稀疏矩阵的coo表示格式
def read_mtx_file(filename):
    with open(filename, "r") as f:
        # 跳过头部注释行
        while True:
            line = f.readline()
            if not line.startswith("%"):
                break
        # 读取矩阵规模信息，并生成稀疏矩阵
        num_rows, num_cols, num_entries = map(int, line.strip().split())
        coo_rows, coo_cols, coo_data = [], [], []
        for i in range(num_entries):
            row, col, data = map(float, f.readline().strip().split())
            coo_rows.append(int(row - 1))   # mtx从1开始编号，转化为从0开始编号
            coo_cols.append(int(col - 1))
            coo_data.append(data)
    return np.array(coo_rows), np.array(coo_cols), np.array(coo_data)

# 计算指定源节点的单源最短路径长度
def sssp(graph, source_node):
    shortest_paths = nx.single_source_dijkstra_path_length(graph, source=source_node)
    return shortest_paths
def bf(graph, source_node):
    shortest_paths = nx.single_source_bellman_ford_path_length(graph,  source=source_node)
    return shortest_paths

# 创建解析器对象并定义命令行参数
parser = argparse.ArgumentParser(description="Calculate shortest path length from source node in mtx graph file")
parser.add_argument("algorithm", metavar="ALGORITHM", type=str, help="Algorithm Dijkstra and Bellman-Ford")
parser.add_argument("input_file", metavar="INPUT_FILE", type=str, help="path to input mtx graph file")
parser.add_argument("output_file", metavar="OUTPUT_FILE", type=str, help="path to output file")
parser.add_argument("source_node", metavar="SOURCE_NODE", type=int, help="source node index (0-based)")

# 解析命令行参数
args = parser.parse_args()

# 获取命令行参数并调用相关函数
coo_rows, coo_cols, coo_data = read_mtx_file(args.input_file)
graph = nx.DiGraph()
for i in range(len(coo_rows)):
    graph.add_edge(coo_rows[i], coo_cols[i], weight=coo_data[i])

if args.algorithm == 'sssp' :
    shortest_paths = sssp(graph, args.source_node)
if args.algorithm == 'bf' :
    shortest_paths = bf(graph, args.source_node)

# 对所有节点按节点编号进行排序
sorted_nodes = sorted(shortest_paths.keys())
with open(args.output_file, "w") as f:
    for node in sorted_nodes:
        length = shortest_paths[node]
        if args.source_node != node:
            f.write("{} {} {:.6f}\n".format(args.source_node, node, length))