#include "access.h"
#include "omp.h"

using namespace std;

struct Matrix
{
    int rows;
    int cols;
    int nnz;
    bool *input;
    int *result;
    int **A;      // 按列压缩
    int *A_entry; // 每列项数
    int **B;      // 按行压缩
    int *B_entry; // 每行项数
    int dim;
    uint64_t entry;
};

void runDawn(Matrix &matrix, string &input_path, string &output_path);
void readgraph(string &input_path, Matrix &matrix);
float dawnSssp(Matrix &matrix, int source);

void readgraph(string &input_path, Matrix &matrix)
{
    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
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
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;
    matrix.dim = dim;

    matrix.A = new int *[matrix.rows];
    matrix.A_entry = new int[matrix.rows];
    matrix.B = new int *[matrix.rows];
    matrix.B_entry = new int[matrix.rows];
    bool **tmp = new bool *[matrix.rows];
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        tmp[i] = new bool[matrix.rows];
        matrix.A_entry[i] = 0;
        matrix.B_entry[i] = 0;
    }
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        ss >> rows >> cols;
        rows--;
        cols--;
        if (rows != cols)
        {
            tmp[rows][cols] = true;
        }
    }
    file.close();

#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        vector<int> tmp_A;
        vector<int> tmp_B;
        tmp_A.clear();
        tmp_B.clear();
        for (int j = 0; j < matrix.cols; j++)
        {
            if (tmp[j][i] == true && i != j)
            {
                tmp_A.push_back(j);
                matrix.A_entry[i]++;
            }
            if (tmp[i][j] == true && i != j)
            {
                tmp_B.push_back(j);
                matrix.B_entry[i]++;
            }
        }
        matrix.A[i] = new int[matrix.A_entry[i]];
        matrix.B[i] = new int[matrix.B_entry[i]];
        for (int j = 0; j < matrix.A_entry[i]; j++)
        {
            matrix.A[i][j] = tmp_A[j];
        }
        for (int j = 0; j < matrix.B_entry[i]; j++)
        {
            matrix.B[i][j] = tmp_B[j];
        }
    }
    cout << "Initialize input matrices" << endl;
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        delete[] tmp[i];
    }
    delete[] tmp;
}

void runDawn(Matrix &matrix, string &input_path, string &output_path)
{
    readgraph(input_path, matrix);

    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return;
    }
    float elapsed_time = 0.0;
    matrix.input = new bool[matrix.rows];
    matrix.result = new int[matrix.rows];
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    for (int i = 0; i < matrix.rows; i++)
    {
        elapsed_time += dawnSssp(matrix, i);
        // 输出结果

        for (int j = 0; j < matrix.rows; j++)
        {
            if (i != j)
                outfile << i << " " << j << " " << matrix.result[j] << endl;
        }
        if (i % (matrix.rows / 100) == 0)
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

float dawnSssp(Matrix &matrix, int source)
{
    int dim = 1;
    int entry = matrix.B_entry[source];
    int entry_last = entry;
    bool *tmp_output = new bool[matrix.rows];
#pragma omp parallel for
    for (int j = 0; j < matrix.rows; j++)
    {
        matrix.input[j] = false;
        matrix.result[j] = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < matrix.B_entry[source]; i++)
    {
        matrix.input[matrix.B[source][i]] = true;
        matrix.result[matrix.B[source][i]] = 1;
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
                if (matrix.input[matrix.A[j][k]] == true)
                {
                    tmp_output[j] = true;
                    break;
                }
            }
        }

#pragma omp parallel for
        for (int j = 0; j < matrix.rows; j++)
        {
            if (matrix.result[j] == 0 && tmp_output[j] == true && j != source)
            {
                matrix.result[j] = dim;
#pragma omp atomic
                entry++;
            }
            matrix.input[j] = tmp_output[j];
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

    Matrix matrix;
    runDawn(matrix, input_path, output_path);
    return 0;
}
