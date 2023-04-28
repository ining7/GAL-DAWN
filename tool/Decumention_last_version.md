# dawn_cpu_v2

```c++
struct Matrix
{
    int rows;
    int cols;
    int nnz;
    bool **input;
    int **dense;
    int *dense_entry;
    bool **result;
    int *entry;
    int dim;
};
```

Matrix struct

````c++
    float runDawn(Matrix &matrix, string &input_path, string &output_path);
    ```
````

Function of running dawn, time complexity O(dnm) and space complexity O(n^2)

````c++
    void readgraph(string &input_path, Matrix &matrix);
    ```
````

Function of reading graph from mtx file

# dawn_gpu

```c++
struct Matrix
{
    int rows;
    int cols;
    int nnz;
    int **dense;
    int *dense_entry;
    int m;
    int n;
    int k;
    int loop;
    bool **result;
    int dim;
};
```

Matrix struct

````c++
    void readgraph(string &input_path, Matrix &matrix);
    ```
````

Function of reading graph from mtx file

````c++
float runDawnGpu(Matrix &matrix, string input_path, string output_path);
    ```
````

Function of running dawn on GPU

````c++
void update_A(**half *&host, Matrix matrix, int rows_start, int rows_end, int n);
    ```
````

Update data of matrix A

````c++
void update_B(**half *&host, Matrix matrix, int cols_start, int cols_end, int n);
    ```
````

Update data of matrix B

````c++
void check_gpu_dense(half *dense, int i_dex, int j_dex, string output_path);
    ```
````

Output matrices copied from GPU to memory for inspecting data and debugging programs.
