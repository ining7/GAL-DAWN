# 大数据跑不动|小数据测试正确 问题主要在 37-42 行 考虑用c++动态编译库
import cupy as cp
import numpy as np
import time
import sys

input_path = ""
N = 0

def writeMatrix(select_direction):
    mtrx = cp.zeros((N, N))
    shortest_distance = np.zeros((N, N), dtype = np.int32)
    with open(input_path, 'r') as f:
        for line in f.readlines():
            a, b = map(int, line.split())
            mtrx[a][b] = 1
            shortest_distance[a, b] = 1
            if select_direction != 0:
                mtrx[b][a] = 1
                shortest_distance[b, a] = 1
    A = cp.sparse.csr_matrix(mtrx)
    B = A      
    return A, B, shortest_distance

def DAWN(A, B, k, k_max, k_diameter, shortest_distance):
    k_last = 0
    dim = 1
    n = N
    while True:
        start_time = time.time()
        print("===", dim)
        dim = dim + 1
        B = B.dot(A)
        y = B.indices
        x = B.tocsc().indices
        x = np.sort(x)
        for idx in range(x.size):
            i = x[idx]
            j = y[idx]
            if i != j and B[i, j] != 0 and shortest_distance[int(i), int(j)] == 0:
                    shortest_distance[int(i), int(j)] = dim
                    k = k + 1
        if k > k_max - 1:
            return shortest_distance
        if k_diameter == dim:
            return shortest_distance
        if k == k_last:
            return shortest_distance
        k_last = k 
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)   

def unweightedPipeline(nodes, select_direction):
    # init
    n = N
    k = 0
    k_max = nodes * (nodes - 1)
    k_diameter = n
    writeMatrix(select_direction)
    A, B, shortest_distance = writeMatrix(select_direction)
    
    # start the timer
    start_time = time.time()
    print("Timing begins")

    shortest_distance = DAWN(A, B, k, k_max, k_diameter, shortest_distance)
    print(shortest_distance)

    end_time = time.time()
    # stop the timer
    print("The total running time of the program is ", end_time - start_time, "s")


def main():
    global input_path 
    input_path = sys.argv[1]
    global N
    N = int(input("Please enter the number of nodes in the graph: "))
    nodes = N
    
    # odd matrix
    if N % 2 != 0:
        N = N + 1
    
    print("[default: unweighted graph]")
    # select_weight = int(input("Please select the type of diagram: 1-Unweighted graph  2-Weighted graph: "))
    select_direction = int(input("Please select the type of diagram: 1-Undirected graph  2-Directed graph: "))
    
    unweightedPipeline(nodes, select_direction)

if __name__ == "__main__":
    main()