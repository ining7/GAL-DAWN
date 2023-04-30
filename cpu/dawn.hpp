#include "access.h"
using namespace std;

class DAWN
{
public:
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
        int thread;
        int interval;
        int stream;
        int block_size;
        bool prinft; // 是否打印结果
        int source;  // 打印的节点
    };

    // v3 and V4
    void COO2CRC(Matrix &matrix, vector<pair<int, int>> &cooMatCol);

    void COO2RCC(Matrix &matrix, vector<pair<int, int>> &cooMatRow);

    void createGraph(string &input_path, Matrix &matrix);

    void readGraph(string &input_path, Matrix &matrix, vector<pair<int, int>> &cooMatCol);

    void runApspV3(Matrix &matrix, string &output_path);

    void runApspV4(Matrix &matrix, string &output_path);

    void runSsspCpu(DAWN::Matrix &matrix, std::string &output_path);

    float sssp_p(DAWN::Matrix &matrix, int source, int *&result);

    float sssp(DAWN::Matrix &matrix, int source, int *&result);

    // big
    void readCRC(Matrix &matrix, string &input_path);

    void readRCC(Matrix &matrix, string &input_path);

    void readGraphBig(string &input_path, string &col_input_path, string &row_input_path, Matrix &matrix);

    // convert
    void createGraphconvert(string &input_path, Matrix &matrix, string &col_output_path, string &row_output_path);

    void COO2RCCconvert(Matrix &matrix, vector<pair<int, int>> &cooMatRow, string &row_output_path);

    void COO2CRCconvert(Matrix &matrix, vector<pair<int, int>> &cooMatCol, string &col_output_path);

    // info
    void infoprint(int entry, int total, int interval, int thread, float elapsed_time);
};

void DAWN::COO2CRC(DAWN::Matrix &matrix, std::vector<pair<int, int>> &cooMatCol)
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

void DAWN::COO2RCC(DAWN::Matrix &matrix, std::vector<pair<int, int>> &cooMatRow)
{
    int row_b = 0;
    int k = 0;
    vector<int> tmp;
    tmp.clear();
    while (k < cooMatRow.size())
    {
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
    }
#pragma omp parallel for
    for (int j = 0; j < matrix.B_entry[row_b]; j++)
    {
        matrix.B[row_b][j] = 0;
        matrix.B[row_b][j] = tmp[j];
    }
}

void DAWN::createGraph(std::string &input_path, DAWN::Matrix &matrix)
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
    file.close();

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.dim = dim;

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
    matrix.nnz = 0;
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.nnz += matrix.A_entry[i];
    }

    cout << "Initialize Input Matrices" << endl;
}

void DAWN::readGraph(std::string &input_path, DAWN::Matrix &matrix, std::vector<pair<int, int>> &cooMatCol)
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

void DAWN::runApspV3(DAWN::Matrix &matrix, std::string &output_path)
{
    float elapsed_time = 0.0;

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    for (int i = 0; i < matrix.rows; i++)
    {
        if (matrix.B_entry[i] == 0)
        {
            infoprint(i, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
            continue;
        }
        elapsed_time += sssp_p(matrix, i, output_path);
        infoprint(i, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
    }
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // Output elapsed time
    std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
}

void DAWN::runApspV4(DAWN::Matrix &matrix, std::string &output_path)
{
    float elapsed_time = 0.0;
    int proEntry = 0;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        if (matrix.B_entry[i] == 0)
        {
#pragma omp critical
            {
                ++proEntry;
                infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
            }
            continue;
        }
        float time_tmp = sssp(matrix, i, output_path);
#pragma omp critical
        {
            elapsed_time += time_tmp;
            ++proEntry;
        }
        infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
    }
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // Output elapsed time and free remaining resources
    std::cout << " Elapsed time: " << elapsed_time / (matrix.thread * 1000) << std::endl;
}

void DAWN::runSsspCpu(DAWN::Matrix &matrix, std::string &output_path)
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return;
    }

    int source = matrix.source;
    if (matrix.B_entry[source] == 0)
    {
        cout << "Source is isolated node, please check" << endl;
        exit(0);
    }

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    float elapsed_time = sssp_p(matrix, source, output_path);

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // Output elapsed time and free remaining resources
    std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
    outfile.close();
}

float DAWN::sssp_p(DAWN::Matrix &matrix, int source, std::string &output_path)
{
    int dim = 1;
    int entry = matrix.B_entry[source];
    int entry_last = entry;
    bool *output = new bool[matrix.rows];
    bool *input = new bool[matrix.rows];
    int *result = new int[matrix.rows];
    int entry_max = matrix.rows - 1;
#pragma omp parallel for
    for (int j = 0; j < matrix.rows; j++)
    {
        input[j] = false;
        output[j] = false;
        result[j] = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < matrix.B_entry[source]; i++)
    {
        input[matrix.B[source][i]] = true;
        output[matrix.B[source][i]] = true;
        result[matrix.B[source][i]] = 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    while (dim < matrix.dim)
    {
        dim++;
#pragma omp parallel for
        for (int j = 0; j < matrix.rows; j++)
        {
            if (matrix.A_entry[j] == 0)
                continue;
            for (int k = 0; k < matrix.A_entry[j]; k++)
            {
                if (input[matrix.A[j][k]] == true)
                {
                    output[j] = true;
                    break;
                }
            }
        }
#pragma omp parallel for
        for (int j = 0; j < matrix.rows; j++)
        {
            if ((result[j] == 0) && (output[j] == true) && (j != source))
            {
                result[j] = dim;
#pragma omp atomic
                entry++;
            }
            input[j] = output[j];
            output[j] = false;
        }
        if ((entry > entry_last) && (entry < matrix.rows - 1))
        {
            entry_last = entry;
            if (entry_last >= entry_max)
                break;
        }
        else
        {
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    matrix.entry += entry_last;

    delete[] output;
    output = nullptr;
    delete[] input;
    input = nullptr;
    // 输出结果
    if ((matrix.prinft) && (source == matrix.source))
    {
        outfile(matrix.rows, result, source, output_path);
    }
    delete[] result;
    result = nullptr;

    return elapsed.count();
}

float DAWN::sssp(DAWN::Matrix &matrix, int source, std::string &output_path)
{
    int dim = 1;
    int entry = matrix.B_entry[source];
    int entry_last = entry;
    bool *output = new bool[matrix.rows];
    bool *input = new bool[matrix.rows];
    int *result = new int[matrix.rows];
    int entry_max = matrix.rows - 1;
    for (int j = 0; j < matrix.rows; j++)
    {
        input[j] = false;
        output[j] = false;
        result[j] = 0;
    }

    for (int i = 0; i < matrix.B_entry[source]; i++)
    {
        input[matrix.B[source][i]] = true;
        output[matrix.B[source][i]] = true;
        result[matrix.B[source][i]] = 1;
    }
    auto start = std::chrono::high_resolution_clock::now();

    while (dim < matrix.dim)
    {
        dim++;

        for (int j = 0; j < matrix.rows; j++)
        {
            if (matrix.A_entry[j] == 0)
                continue;
            for (int k = 0; k < matrix.A_entry[j]; k++)
            {
                if (input[matrix.A[j][k]] == true)
                {
                    output[j] = true;
                    break;
                }
            }
        }
        for (int j = 0; j < matrix.rows; j++)
        {
            if ((result[j] == 0) && (output[j] == true) && (j != source))
            {
                result[j] = dim;
                entry++;
            }
            input[j] = output[j];
            output[j] = false;
        }
        if ((entry > entry_last) && (entry < matrix.rows - 1))
        {
            entry_last = entry;
            if (entry_last >= entry_max)
                break;
        }
        else
        {
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    matrix.entry += entry_last;

    delete[] output;
    output = nullptr;
    delete[] input;
    input = nullptr;
    // 输出结果
    if ((matrix.prinft) && (source == matrix.source))
    {
        outfile(matrix.rows, result, source, output_path);
    }
    delete[] result;
    result = nullptr;

    return elapsed.count();
}

void DAWN::outfile(int n, int *result, int source, std::string &output_path)
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return;
    }
    for (int j = 0; j < n; j++)
    {
        if ((source != j) && (result[j] > 0))
            outfile << source << " " << j << " " << result[j] << endl;
    }
    outfile.close();
}

void DAWN::readCRC(DAWN::Matrix &matrix, std::string &input_path)
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

void DAWN::readRCC(DAWN::Matrix &matrix, std::string &input_path)
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

void DAWN::readGraphBig(std::string &input_path, std::string &col_input_path, std::string &row_input_path, DAWN::Matrix &matrix)
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
    matrix.dim = dim;

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

void DAWN::createGraphconvert(std::string &input_path, DAWN::Matrix &matrix, std::string &col_output_path, std::string &row_output_path)
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
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.dim = dim;

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
    COO2CRCconvert(matrix, cooMatCol, col_output_path);
    COO2RCCconvert(matrix, cooMatRow, row_output_path);
    cout << "Initialize Input Matrices" << endl;
}

void DAWN::COO2CRCconvert(DAWN::Matrix &matrix, std::vector<pair<int, int>> &cooMatCol, std::string &col_output_path)
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
        infoprint(col_a, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
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

void DAWN::COO2RCCconvert(DAWN::Matrix &matrix, std::vector<pair<int, int>> &cooMatRow, std::string &row_output_path)
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
        infoprint(row_b, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
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

void DAWN::infoprint(int entry, int total, int interval, int thread, float elapsed_time)
{
    if (entry % (total / interval) == 0)
    {
        float completion_percentage =
            static_cast<float>(entry * 100.0f) / static_cast<float>(total);
        std::cout << "Progress: " << completion_percentage << "%" << std::endl;
        std::cout << "Elapsed Time :" << elapsed_time / (thread * 1000) << " s" << std::endl;
    }
}
