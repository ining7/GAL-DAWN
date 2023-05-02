#include "access.h"
#include "omp.h"

using namespace std;

struct Matrix
{
    int rows;
    int cols;
    uint64_t nnz;
    int **A;      // 按列压缩
    int *A_entry; // 每列项数
    int **B;      // 按行压缩
    int *B_entry; // 每行项数
    int dim;
    uint64_t entry;
};

void runDawn(Matrix &matrix, string &input_path, string &output_path);
void createGraph(string &input_path, Matrix &matrix);
void readGraph(string &input_path, Matrix &matrix, vector<pair<int, int>> &cooMatCol);
void COO2CRC(Matrix &matrix, vector<pair<int, int>> &cooMatCol);
void COO2RCC(Matrix &matrix, vector<pair<int, int>> &cooMatRow);
float dawnSssp(Matrix &matrix, int source);

void COO2CRC(Matrix &matrix, vector<pair<int, int>> &cooMatCol)
{
    int col_a = 0;
    int k = 0;
    vector<int> tmp;
    tmp.clear();
    while (k < cooMatCol.size())
    {
        if (matrix.A_entry[col_a] != 0)
        {
            if (cooMatCol[k].second == col_a)
            {
                tmp.push_back(cooMatCol[k].first);
                k++;
            }
            else
            {
#pragma omp parallel for
                for (int j = 0; j < matrix.A_entry[col_a]; j++)
                {
                    matrix.A[col_a][j] = tmp[j];
                }
                tmp.clear();
                col_a++;
            }
        }
        else
        {
            col_a++;
        }
    }
#pragma omp parallel for
    for (int j = 0; j < matrix.A_entry[col_a]; j++)
    {
        matrix.A[col_a][j] = tmp[j];
    }
}

void COO2RCC(Matrix &matrix, vector<pair<int, int>> &cooMatRow)
{
    int row_b = 0;
    int k = 0;
    vector<int> tmp;
    tmp.clear();
    while (k < cooMatRow.size())
    {
        if (matrix.A_entry[row_b] != 0)
        {
            if (cooMatRow[k].first == row_b)
            {
                tmp.push_back(cooMatRow[k].second);
                k++;
            }
            else
            {
#pragma omp parallel for
                for (int j = 0; j < matrix.B_entry[row_b]; j++)
                {
                    matrix.B[row_b][j] = tmp[j];
                }
                tmp.clear();
                row_b++;
            }
        }
        else
        {
            row_b++;
        }
    }
#pragma omp parallel for
    for (int j = 0; j < matrix.B_entry[row_b]; j++)
    {
        matrix.B[row_b][j] = 0;
        matrix.B[row_b][j] = tmp[j];
    }
}

void readCRC(Matrix &matrix, string &input_path)
{
    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }
    matrix.A = new int *[matrix.rows];
    string line;
    int rows = 0, cols = 0, k = 0;
    while (getline(file, line))
    {
        stringstream ss(line);
        ss >> cols;
        if (cols == 0)
        {
            matrix.A_entry[rows] = 0;
            rows++;
            k = 0;
            continue;
        }
        matrix.A_entry[rows] = cols;
        matrix.A[rows] = new int[cols];
        for (int j = 0; j < cols; j++)
        {
            ss >> matrix.A[rows][k++];
        }
        rows++;
        k = 0;
    }
}
void readRCC(Matrix &matrix, string &input_path)
{
    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }
    matrix.B = new int *[matrix.rows];
    string line;
    int rows = 0, cols = 0, k = 0;
    while (getline(file, line))
    {
        stringstream ss(line);
        ss >> cols;
        if (cols == 0)
        {
            matrix.B_entry[rows] = 0;
            rows++;
            k = 0;
            continue;
        }
        matrix.B_entry[rows] = cols;
        matrix.B[rows] = new int[cols];
        for (int j = 0; j < cols; j++)
        {
            ss >> matrix.B[rows][k++];
        }
        rows++;
        k = 0;
    }
}

void readGraph(string &input_path, Matrix &matrix, vector<pair<int, int>> &cooMatCol)
{
    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }
    std::string line;

    int rows, cols;
    while (std::getline(file, line))
    {
        if (line[0] == '%')
            continue;
        std::stringstream ss(line);
        ss >> rows >> cols;
        rows--;
        cols--;
        if (rows != cols)
        {
            cooMatCol.push_back({rows, cols});
            matrix.A_entry[cols]++;
            matrix.B_entry[rows]++;
            matrix.entry++;
        }
    }
    file.close();

    matrix.nnz = matrix.entry;
    cout << "nnz: " << matrix.nnz << endl;
}

void createGraph(string &input_path, Matrix &matrix)
{

    matrix.A = new int *[matrix.rows];
    matrix.A_entry = new int[matrix.rows];
    matrix.B = new int *[matrix.rows];
    matrix.B_entry = new int[matrix.rows];
    matrix.entry = 0;
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.A_entry[i] = 0;
        matrix.B_entry[i] = 0;
    }

    vector<pair<int, int>> cooMatCol;
    cout << "Read Input Graph" << endl;
    readGraph(input_path, matrix, cooMatCol);

    cout << "Create Transport Graph" << endl;
    vector<pair<int, int>> cooMatRow = cooMatCol;
    sort(cooMatRow.begin(), cooMatRow.end());

    cout << "Create Input Matrices" << endl;
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.A[i] = new int[matrix.A_entry[i]];
        matrix.B[i] = new int[matrix.B_entry[i]];
    }
    COO2CRC(matrix, cooMatCol);
    COO2RCC(matrix, cooMatRow);
    cout << "Initialize Input Matrices" << endl;
}

void runDawn(Matrix &matrix, string &input_path, string &output_path)
{
    createGraph(input_path, matrix);

    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return;
    }
    float elapsed_time = 0.0;
    int proEntry = 0;
    int thread = 20;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        float time_tmp = 0.0f;
        if (matrix.B_entry[i] == 0)
            continue;
        time_tmp = dawnSssp(matrix, i);
#pragma omp critical
        {
            elapsed_time += time_tmp;
            proEntry++;
        }
        // 输出结果

        if (proEntry % (matrix.rows / 100) == 0)
        {
            float completion_percentage =
                static_cast<float>(proEntry * 100.0f) / static_cast<float>(matrix.rows);
            std::cout << "Progress: " << completion_percentage << "%" << std::endl;
            std::cout << "Elapsed Time :" << elapsed_time / (thread * 1000) << " s" << std::endl;
        }
    }
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // Output elapsed time and free remaining resources
    std::cout << " Elapsed time: " << elapsed_time / (thread * 1000) << std::endl;
    outfile.close();
}

float dawnSssp(Matrix &matrix, int source)
{
    int dim = 1;
    int entry = matrix.B_entry[source];
    int entry_last = entry;
    bool *tmp_output = new bool[matrix.rows];
    bool *input = new bool[matrix.rows];
    int *result = new int[matrix.rows];

    for (int j = 0; j < matrix.rows; j++)
    {
        input[j] = false;
        result[j] = 0;
    }

    for (int i = 0; i < matrix.B_entry[source]; i++)
    {
        input[matrix.B[source][i]] = true;
        result[matrix.B[source][i]] = 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    while (dim < matrix.dim)
    {
        dim++;

        for (int j = 0; j < matrix.rows; j++)
        {
            for (int k = 0; k < matrix.A_entry[j]; k++)
            {
                if (input[matrix.A[j][k]] == true)
                {
                    tmp_output[j] = true;
                    break;
                }
            }
        }

        for (int j = 0; j < matrix.rows; j++)
        {
            if (result[j] == 0 && tmp_output[j] == true && j != source)
            {
                result[j] = dim;
                entry++;
            }
            input[j] = tmp_output[j];
            tmp_output[j] = false;
        }

        if (entry > entry_last)
        {
            entry_last = entry;
            if (entry >= matrix.rows - 1) // entry = matrix.rows - 1意味着向量填满，无下一轮
                break;
        }
        else // entry = entry_last表示没有发现新的路径，也不再进行下一轮
            break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    matrix.entry += entry_last;

    // for (int j = 0; j < matrix.rows; j++)
    // {
    //     if (i != j)
    //         outfile << i << " " << j << " " << result[j] << endl;
    // }

    delete[] tmp_output;
    tmp_output = nullptr;
    delete[] result;
    result = nullptr;
    delete[] input;
    input = nullptr;

    return elapsed.count();
}

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string output_path = argv[2];

    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return 0;
    }

    std::string line;
    int rows, cols, nnz, dim;
    while (std::getline(file, line))
    {
        if (line[0] == '%')
            continue;

        std::stringstream ss(line);
        ss >> rows >> cols >> nnz >> dim;
        break;
    }
    cout << rows << " " << cols << " " << nnz << " " << dim << endl;

    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.dim = dim;

    runDawn(matrix, input_path, output_path);
    return 0;
}