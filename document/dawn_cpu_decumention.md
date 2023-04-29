## Matrix class

### Function

1. ```c++
   Matrix() {}
   Matrix(string _g_type, string _device, AlgoType _algo_type, int _row, int _col):
   ```

   <!-- ​	构造函数：初始化 Matrix 类的实例，根据类型和设备分配内存。

   ​	最短路的答案统一保存在数组`g_dense_gpu`中。 -->

   ​ Constructor: Initialize an instance of the Matrix class, and allocate memory according to the type and device.

   ​ The shortest path answers are uniformly stored in the array `g_dense_gpu`.

2. ```c++
   ~Matrix()
   ```

   <!-- ​ 析构函数 -->

   destructor

3. ```c++
   inline void setValueGpuDense(int i, int j, int value)
   ```

   <!-- ​ 在 GPU 稠密矩阵的计算中，将下标为(i,j)的值设为 value -->

   ​ In the calculation of GPU dense matrix, set the value subscripted as (i,j) as value

4. ```c++
   inline void setValueCpuSparse(int i, int j, long long value)
   ```

   <!-- ​ 在 CPU 稠密矩阵的计算中，将下标为(i,j)的值设为 value -->

   ​ In the calculation of CPU dense matrix, set the value subscripted as (i,j) to value

5. ```C++
      inline int getValueGpuDense(int i, int j, bool flag_press)
   ```

      <!-- ​ 在 GPU 稠密矩阵的计算中，获取下标为(i,j)的值 -->

   In the calculation of GPU dense matrix, get the value subscripted as (i,j)

   <!-- - flag_press 为 true 的时候是由压位后的矩阵进行调用
   - flag_press 为 false 的时候是由未压位的矩阵进行调用 -->

   - When flag_press is true, it is called by the pressed matrix
   - When flag_press is false, it is called by the unpressed matrix

6. ```c++
   inline int getValueCpuSparse(int i, int j)
   ```

   <!-- ​ 在 CPU 稀疏矩阵的计算中，获取下标为(i,j)的值 -->

   In the calculation of the CPU sparse matrix, get the value with the subscript (i,j)

7. ```c++
   inline int getValueCpuDense(int i, int j)
   ```

   <!-- ​ 在 CPU 稠密矩阵的计算中，获取下标为(i,j)的值 -->

   ​ In the calculation of CPU dense matrix, get the value with subscript (i,j)

8. ```c++
   void readData(bool* in)
   ```

   <!-- ​ 对不同的设备与矩阵类型进行数据的写入，用于读数据 -->

   Write data to different devices and matrix types for reading data

9. ```c++
   void pressData(bool *in)
   ```

   <!-- ​ 进行压位操作，将传入的 bool 数组压成 int32 的类型，压位后的值直接写入 g_dense_gpu 数组内 -->

   ​ Carry out compression operation, compress the incoming bool array into int32 type, and write the compressed value directly into the g_dense_gpu array

10. ```c++
    void printMatrix()
    ```

    <!-- ​ 输出不同类型的矩阵，用于调试，在算法中没有任何实质性作用 -->

    Output different types of matrices, for debugging, without any substantive role in the algorithm

11. ```c++
    void denseCpuMultiplication(Matrix* A)
    ```

    <!-- ​ CPU 上稠密矩阵乘法，通过一个 Matrix 类进行调用，this 的矩阵与 A 矩阵进行乘法，答案写入 this 对应矩阵数组内 -->

    Dense matrix multiplication on the CPU is called through a Matrix class, the matrix of this is multiplied with the matrix A, and the answer is written into the matrix array corresponding to this

12. ```c++
    void sparseCpuMultiplication(Matrix* A)
    ```

    <!-- ​ CPU 上稀疏矩阵乘法，通过一个 Matrix 类进行调用，this 的矩阵与 A 矩阵进行乘法，答案写入 this 对应矩阵数组内 -->

    ​ Sparse matrix multiplication on the CPU is called through a Matrix class, the matrix of this is multiplied with the matrix A, and the answer is written into the matrix array corresponding to this

13. ```c++
    void denseGpuMultiplication(Matrix* A)
    ```

    <!-- ​ GPU 上稠密矩阵乘法，通过一个 Matrix 类进行调用，this 的矩阵与 A 矩阵进行乘法，答案写入 this 对应矩阵数组内，其中对`denseIntMultiplicationKernel`和`change`函数进行了调用 -->

    ​ Dense matrix multiplication on the GPU is called through a Matrix class, the matrix of this is multiplied with the A matrix, and the answer is written into the matrix array corresponding to this, in which the `denseIntMultiplicationKernel` and `change` functions are called

### 变量

<!-- 1. string g_type：用于表示图的类型，有 sparse 和 dense -->

1. string g_type：The type used to represent the graph, there are sparse and dense

<!-- 2. string device：用于表示设备的类型，有 cpu 和 gpu -->

2. string device: used to indicate the type of device, including cpu and gpu

<!-- 3. AlgoType algo_type：用于表示使用的算法，此处默认为 dawn -->

3. AlgoType algo_type: Used to indicate the algorithm used, here the default is dawn

<!-- 4. long long row：表示这个 Matrix 的行数 -->

4. long long row: Indicates the number of rows in this Matrix

<!-- 5. long long col：表示这个 Matrix 的列数 -->

5. long long col: Indicates the number of columns of this Matrix

<!-- 6. vector<pair<int, long long>> g_sparse_cpu[kmaxRow]：用于存放 cpu 上的稀疏矩阵 -->

6. vector<pair<int, long long>> g_sparse_cpu[kmaxRow]: used to store sparse matrix on cpu

<!-- 7. vector<bitset<kmaxBitsetThou>> g_dense_cpu_thou;

   vector<bitset<kmaxBitsetTenThou>>g_dense_cpu_ten_thou;

   vector<bitset<kmaxBitsetHunThou>> g_dense_cpu_hun_thou;

   ​ 均用于保存 cpu 上的稠密矩阵，由于 bitset 必须在编译前指定 size，故设定不同的大小 -->

7.  vector<bitset<kmaxBitsetThou>> g_dense_cpu_thou;

    vector<bitset<kmaxBitsetTenThou>>g_dense_cpu_ten_thou;

    vector<bitset<kmaxBitsetHunThou>> g_dense_cpu_hun_thou;

    ​ Both are used to save the dense matrix on the cpu. Since the bitset must specify the size before compiling, set different sizes

<!-- 8.  long long stride：表示当前矩阵的宽，便于进行压位后的操作，需要与 col 进行区分 -->

8. long long stride: Indicates the width of the current matrix, which is convenient for operations after pressing, and needs to be distinguished from col

<!-- 9. unsigned int \*g_dense_gpu： 存放 gpu 上的稠密矩阵，由于数据类型的优势，也用于各种类型的最短路最终结果的存放数组 -->

9. unsigned int \*g_dense_gpu: Store the dense matrix on the gpu, due to the advantages of the data type, it is also used for storing arrays of various types of shortest path final results

## Graph

### 函数

<!-- 1. ```c++
   Graph(string _g_type, string _device, int _node_num, AlgoType _algo_type):
   		node_num(_node_num), algo_type(_algo_type)
   ```

   ​ 构造函数

2. ```c++
   ~Graph()
   ```

   ​ 析构函数

3. ```c++
   static AlgoType parseAlgo(string file_name)
   ```

   ​ 通过传入的 file_name 判断本次算法需要使用的算法类型，此处默认返回 dawn 算法

4. ```c++
   void readMap(string file_name, string device, string g_type, string random_flag)
   ```

   ​ 从文件中读取图数据，并对 A ,B,最短路答案矩阵进行初始化

5. ```c++
   void updateShortestPath(int dim, string g_type, string device)
   ```

   ​ 更新最短路径，用于算法中答案矩阵的更新

6. ```c++
   void runDawn(string g_type, string device)
   ```

   ​ dawn 算法的核心部分

7. ```c++
   void runDij() {}
   void runSpfa() {}
   ```

   ​ 本次算法中未被使用的 Dij 和 Spfa 算法的接口

8. ```c++
   void runShortestPath(string g_type, string device)
   ```

   ​ 根据 algo_type 类型调用对应的算法

9. ```c++
   void saveRes(string file_name)
   ```

   ​ 将最短路答案写入指定的 file_name 路径的文件夹中 -->

1.  ```c++
    Graph(string _g_type, string _device, int _node_num, AlgoType _algo_type):
    node_num(_node_num), algo_type(_algo_type)
    ```

    ```

    ​ Constructor

    ```

2.  ```c++
    ~Graph()
    ```

    ​ Destructor

3.  ```c++
    static AlgoType parseAlgo(string file_name)
    ```

    ​ Determine the type of algorithm to be used in this algorithm by passing in the file_name, here returns the dawn algorithm by default

4.  ```c++
    void readMap(string file_name, string device, string g_type, string random_flag)
    ```

    ​ Read the graph data from the file, and initialize A, B, and the shortest path answer matrix

5.  ```c++
    void updateShortestPath(int dim, string g_type, string device)
    ```

    ​ Update the shortest path for updating the answer matrix in the algorithm

6.  ```c++
    void runDawn(string g_type, string device)
    ```

    ​ The core part of the dawn algorithm

7.  ```c++
    void runDij() {}
    void runSpfa() {}
    ```

    ​ Interfaces of Dij and Spfa algorithms that are not used in this algorithm

8.  ```c++
    void runShortestPath(string g_type, string device)
    ```

    ​ Call the corresponding algorithm according to the type of algo_type

9.  ```c++
    void saveRes(string file_name)
    ```

    ​ Write the shortest path answer to the folder specified by the file_name path

### 变量

<!-- 1. AlgoType algo_type：进行此图最短路求解的算法类型
2. Matrix \*A：矩阵 A
3. Matrix \*B：矩阵 B
4. Matrix \*res：答案矩阵
5. long long k：发现的路径总数
6. long long k_max：可能存在的路径最大数量，为 n\*(n-1)
7. long long k_diameter：图的直径
8. long long node_num：图中节点的数量 -->

1. AlgoType algo_type: The algorithm type for the shortest path solution of this graph
2. Matrix \*A: Matrix A
3. Matrix \*B: matrix B
4. Matrix \*res: answer matrix
5. long long k: total number of paths found
6. long long k_max: the maximum number of possible paths, n\*(n-1)
7. long long k_diameter: the diameter of the graph
8. long long node_num: the number of nodes in the graph

````c++
// Graph* readNeighboorMatrix(string file_name,
// 		string g_type, string device, string random_flag, string _node_num)
// ```

// - 将节点数进行了 string 到 long long 的转换
// - 获取算法类型
// - 新建一个图，用于本次算法
// - 从文件中读图

// ```c++
// int main(int argc, char *argv[])
// ```

// - main 函数，从指令中获取一些需要的信息
// - 新建图
// - 运行最短路算法
// - 保存最短路结果

// ```c++
// __global__ void denseIntMultiplicationKernel(long long N, long long M, long long K, unsigned int* A, unsigned int* B, bool* res)
// ```

// ​ kernel 函数，用于进行 gpu 上的稠密矩阵乘法

// ​ N\*M 的矩阵 A 　于　Ｍ＊Ｋ的矩阵 B 　进行乘法，存放到ｂｏｏｌ矩阵ｒｅｓ中

// ```c++
// void change(long long N, long long M, long long K, bool C[], unsigned int* A)
// ```

// ​ 进行 bool 矩阵压 int 的操作，row 变成 row/32

// ​ N*M 的 bool 矩阵 C 压成 M * K 的矩阵 A

Graph* readNeighboorMatrix(string file_name,
string g_type, string device, string random_flag, string _node_num)
````

- Converted the number of nodes from string to long long
- Get algorithm type
- Create a new graph for this algorithm
- read image from file

```c++
int main(int argc, char *argv[])
```

- main function, get some needed information from the instruction
- New graph
- Run the shortest path algorithm
- save the shortest path result

```c++
__global__ void denseIntMultiplicationKernel(long long N, long long M, long long K, unsigned int* A, unsigned int* B, bool* res)
```

​ Kernel function for dense matrix multiplication on gpu

​ The matrix A of N\*M is multiplied by the matrix B of M\*K and stored in the bool matrix res

```c++
void change(long long N, long long M, long long K, bool C[], unsigned int* A)
```

​ Carry out the operation of bool matrix pressing int, row becomes row/32

​ N*M bool matrix C is compressed into M * K matrix A
