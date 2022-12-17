import networkx as nx             
import numpy as np
import time

# 待处理数据集路径
edgeListPath = "in.txt"

# ===== 开始计时
start=time.time()

# 有向图
G0=nx.DiGraph()
G0=nx.read_edgelist(edgeListPath,create_using=nx.DiGraph)
print(nx.adjacency_matrix(G0))

# 无向图
G1=nx.Graph()
G1=nx.read_edgelist(edgeListPath)
print(nx.adjacency_matrix(G1))

# 生成节点数为500，边概率为0.2的随机无向图
G2 = nx.random_graphs.erdos_renyi_graph(500, 0.2)
print(nx.adjacency_matrix(G2))

# 生成节点数为500，节点度数为10的随机无向图
G3 = nx.random_graphs.barabasi_albert_graph(500, 10)
print(nx.adjacency_matrix(G3))

# ===== 停止计时
end=time.time()

print('Data preprocessing time: %s Seconds'%(end-start))