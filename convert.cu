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

void createGraph(string &input_path, Matrix &matrix, string &col_output_path, string &row_output_path);
void readGraph(string &input_path, Matrix &matrix, vector<pair<int, int>> &cooMatCol);
void COO2RCC(Matrix &matrix, vector<pair<int, int>> &cooMatRow, string &row_output_path);
void COO2CRC(Matrix &matrix, vector<pair<int, int>> &cooMatCol, string &col_output_path);

void COO2CRC(Matrix &matrix, vector<pair<int, int>> &cooMatCol, string &col_output_path)
{
    std::ofstream outfile(col_output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << col_output_path << std::endl;
        return;
    }

    int col_a = 0;
    int k = 0;
    vector<int> tmp;
    tmp.clear();
    float elapsed_time = 0.0;
    int thread = 20;
    while (k < cooMatCol.size())
    {
        auto start = std::chrono::high_resolution_clock::now();
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
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        elapsed_time += elapsed.count();
        if (col_a % (matrix.rows / 100) == 0)
        {
            float completion_percentage =
                static_cast<float>(col_a * 100.0f) / static_cast<float>(matrix.rows);
            std::cout << "Progress: " << completion_percentage << "%" << std::endl;
            std::cout << "Elapsed Time :" << elapsed_time / (thread * 1000) << " s" << std::endl;
        }
    }
#pragma omp parallel for
    for (int j = 0; j < matrix.A_entry[col_a]; j++)
    {
        matrix.A[col_a][j] = tmp[j];
    }
    outfile << matrix.rows << " " << matrix.cols << " " << matrix.nnz << " " << matrix.dim << " " << endl;
    for (int i = 0; i < matrix.rows; i++)
    {
        outfile << matrix.A_entry[i] << " ";
        for (int j = 0; j < matrix.A_entry[i]; j++)
        {
            outfile << matrix.A[i][j] << " ";
        }
        outfile << endl;
    }
}

void COO2RCC(Matrix &matrix, vector<pair<int, int>> &cooMatRow, string &row_output_path)
{
    std::ofstream outfile(row_output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << row_output_path << std::endl;
        return;
    }
    cout << "create B" << endl;

    int row_b = 0;
    int k = 0;
    vector<int> tmp;
    tmp.clear();
    float elapsed_time = 0.0;
    int thread = 20;
    while (k < cooMatRow.size())
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (matrix.B_entry[row_b] != 0)
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
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        elapsed_time += elapsed.count();
        if (row_b % (matrix.rows / 100) == 0)
        {
            float completion_percentage =
                static_cast<float>(row_b * 100.0f) / static_cast<float>(matrix.rows);
            std::cout << "Progress: " << completion_percentage << "%" << std::endl;
            std::cout << "Elapsed Time :" << elapsed_time / (thread * 1000) << " s" << std::endl;
        }
    }
    cout << "compress B" << endl;
#pragma omp parallel for
    for (int j = 0; j < matrix.B_entry[row_b]; j++)
    {
        matrix.B[row_b][j] = tmp[j];
    }
    outfile << matrix.rows << " " << matrix.cols << " " << matrix.nnz << " " << matrix.dim << " " << endl;
    for (int i = 0; i < matrix.rows; i++)
    {
        outfile << matrix.B_entry[i] << " ";
        for (int j = 0; j < matrix.B_entry[i]; j++)
        {
            outfile << matrix.B[i][j] << " ";
        }
        outfile << endl;
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
            //  matrix.A_entry[cols]++;
            matrix.B_entry[rows]++;
            matrix.entry++;
        }
    }
    file.close();

    matrix.nnz = matrix.entry;
    cout << "nnz: " << matrix.nnz << endl;
}

void createGraph(string &input_path, Matrix &matrix, string &col_output_path, string &row_output_path)
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
    COO2CRC(matrix, cooMatCol, col_output_path);
    COO2RCC(matrix, cooMatRow, row_output_path);
    cout << "Initialize Input Matrices" << endl;
}

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string col_output_path = argv[2];
    string row_output_path = argv[3];

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

    createGraph(input_path, matrix, col_output_path, row_output_path);
    return 0;
}
