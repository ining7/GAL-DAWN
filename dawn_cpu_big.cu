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

void readCRC(Matrix &matrix, string &input_path);
void readRCC(Matrix &matrix, string &input_path);
void runDawn(Matrix &matrix, string &output_path);
void readGraphBig(string &col_input_path, string &row_input_path, Matrix &matrix);
float dawnSsspP(Matrix &matrix, int source);

void readCRC(Matrix &matrix, string &input_path)
{
    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }

    matrix.A = new int *[matrix.rows];
    matrix.A_entry = new int[matrix.rows];
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.A_entry[i] = 0;
    }

    string line;
    int rows = 0, cols = 0, k = 0;
    while (getline(file, line))
    {
        stringstream ss(line);
        ss >> cols;
        if (cols == matrix.rows)
        {
            continue;
        }
        if ((cols == 0) || (cols == matrix.rows))
        {
            if (cols == 0)
                rows++;
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
    matrix.B_entry = new int[matrix.rows];
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.B_entry[i] = 0;
    }
    string line;
    int rows = 0, cols = 0, k = 0;
    while (getline(file, line))
    {
        stringstream ss(line);
        ss >> cols;
        if (cols == matrix.rows)
        {
            continue;
        }
        if ((cols == 0) || (cols == matrix.rows))
        {
            if (cols == 0)
                rows++;
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

void readGraphBig(string &col_input_path, string &row_input_path, Matrix &matrix)
{
    cout << "readCRC" << endl;
    readCRC(matrix, col_input_path);
    cout << "readRCC" << endl;
    readRCC(matrix, row_input_path);
    matrix.nnz = 0;
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.nnz += matrix.A_entry[i];
    }
    cout << "nnz: " << matrix.nnz << endl;
    matrix.nnz = 0;
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.nnz += matrix.B_entry[i];
    }
    matrix.entry = matrix.nnz;
    cout << "nnz: " << matrix.nnz << endl;
    cout << "Initialize Input Matrices" << endl;
}

void runDawn(Matrix &matrix, string &output_path)
{

    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return;
    }
    float elapsed_time = 0.0;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    for (int i = 0; i < matrix.rows; i++)
    {
        float time_tmp = 0.0f;
        if (matrix.B_entry[i] == 0)
            continue;
        time_tmp = dawnSsspP(matrix, i);

        elapsed_time += time_tmp;
        // 输出结果

        if (i % (matrix.rows / 10000) == 0)
        {
            float completion_percentage =
                static_cast<float>(i * 100.0f) / static_cast<float>(matrix.rows);
            std::cout << "Progress: " << completion_percentage << "%" << std::endl;
            std::cout << "Elapsed Time :" << elapsed_time / 1000 << " s" << std::endl;
        }
    }
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // Output elapsed time and free remaining resources
    std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
    outfile.close();
}

float dawnSsspP(Matrix &matrix, int source)
{
    int dim = 1;
    int entry = matrix.B_entry[source];
    int entry_last = entry;
    bool *tmp_output = new bool[matrix.rows];
    bool *input = new bool[matrix.rows];
    int *result = new int[matrix.rows];

#pragma omp parallel for
    for (int j = 0; j < matrix.rows; j++)
    {
        input[j] = false;
        result[j] = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < matrix.B_entry[source]; i++)
    {
        input[matrix.B[source][i]] = true;
        result[matrix.B[source][i]] = 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    while (dim < matrix.dim)
    {
        dim++;
// cout << " dim: " << dim << endl;
#pragma omp parallel for
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

#pragma omp parallel for
        for (int j = 0; j < matrix.rows; j++)
        {
            if (result[j] == 0 && tmp_output[j] == true && j != source)
            {
                result[j] = dim;
#pragma omp atomic
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
    string col_input_path = argv[1];
    string row_input_path = argv[2];
    string output_path = argv[3];

    std::ifstream file(col_input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << col_input_path << std::endl;
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
    readGraphBig(col_input_path, row_input_path, matrix);
    runDawn(matrix, output_path);
    return 0;
}
