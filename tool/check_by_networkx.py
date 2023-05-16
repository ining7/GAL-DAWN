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
        num_rows, num_cols, num_entries, dim = map(int, line.strip().split())
        coo_rows, coo_cols = [], []
        for i in range(num_entries):
            row, col = map(int, f.readline().strip().split())
            coo_rows.append(row - 1)   # mtx从1开始编号，转化为从0开始编号
            coo_cols.append(col - 1)
    return np.array(coo_rows), np.array(coo_cols)

# 计算指定源节点的单源最短路径长度
def shortest_path_length(graph, source_node):
    shortest_paths = nx.single_source_shortest_path_length(graph, source=source_node)
    return shortest_paths

# 创建解析器对象并定义命令行参数
parser = argparse.ArgumentParser(description="Calculate shortest path length from source node in mtx graph file")
parser.add_argument("input_file", metavar="INPUT_FILE", type=str, help="path to input mtx graph file")
parser.add_argument("output_file", metavar="OUTPUT_FILE", type=str, help="path to output file")
parser.add_argument("source_node", metavar="SOURCE_NODE", type=int, help="source node index (0-based)")

# 解析命令行参数
args = parser.parse_args()

# 获取命令行参数并调用相关函数
coo_rows, coo_cols = read_mtx_file(args.input_file)
graph = nx.DiGraph()
for i in range(len(coo_rows)):
    graph.add_edge(coo_rows[i], coo_cols[i])
shortest_paths = shortest_path_length(graph, args.source_node)
# 对所有节点按节点编号进行排序
sorted_nodes = sorted(shortest_paths.keys())
with open(args.output_file, "w") as f:
    for node in sorted_nodes:
        length = shortest_paths[node]
        f.write("{} {} {}\n".format(args.source_node, node, length))

