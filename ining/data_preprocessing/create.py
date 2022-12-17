import networkx as nx     
import random          
import numpy as np
import time
import cugraph as cu
# conda activate rapids-22.10

# 开始计时
start=time.time()

G3 = nx.random_graphs.erdos_renyi_graph(500, 0.1)

# 写入处理文件
a=np.array(nx.adjacency_matrix(G3).todense())
f = open('in.txt','w')
for i in range(len(a)):
    for j in range(len(a[0])):
        if a[i][j] != 0:
            f.write(str(i) + ' ' + str(j) + '\n')
f.close()

# 停止计时
end=time.time()

print('Data preprocessing time: %s Seconds'%(end-start))