#include "access.h"
using namespace std;

__global__ void vecMatOpe(int rows, int *d_A_entry, int *d_A, bool *input, bool *output, int *result, int source, int dim, int *d_entry);

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

    float sssp_p(DAWN::Matrix &matrix, int source, std::string &output_path);

    float sssp(DAWN::Matrix &matrix, int source, std::string &output_path);

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

    void runApspGpu(DAWN::Matrix &matrix, std::string &output_path);

    void runSsspGpu(DAWN::Matrix &matrix, std::string &output_path);

    float sssp_gpu(DAWN::Matrix &matrix, int source, cudaStream_t streams, int *d_A_entry, int *d_A, int *&result);
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
    uint32_t dim = 1;
    uint32_t entry = matrix.B_entry[source];
    uint32_t entry_last = entry;
    bool *output = new bool[matrix.rows];
    bool *input = new bool[matrix.rows];
    int *result = new int[matrix.rows];
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
            if ((result[j] == 0) && (output[j] == true) && (source != j))
            {
                result[j] = dim;
#pragma omp atomic
                entry++;
            }
            input[j] = output[j];
            output[j] = false;
        }
        if ((entry > entry_last) && (entry < matrix.rows))
        {
            entry_last = entry;
            if (entry_last >= matrix.rows - 1) // entry = matrix.rows - 1意味着向量填满，无下一轮
                break;
        }
        else // 如果没有新的最短路径产生，则退出循环
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
        std::ofstream outfile(output_path);
        if (!outfile.is_open())
        {
            std::cerr << "Error opening file " << output_path << std::endl;
            return 0.0;
        }
        for (int j = 0; j < matrix.rows; j++)
        {
            if ((source != j) && (result[j] != 0))
                outfile << source << " " << j << " " << result[j] << endl;
        }
        outfile.close();
    }
    delete[] result;
    result = nullptr;

    return elapsed.count();
}

float DAWN::sssp(DAWN::Matrix &matrix, int source, std::string &output_path)
{
    uint32_t dim = 1;
    uint32_t entry = matrix.B_entry[source];
    uint32_t entry_last = entry;
    bool *output = new bool[matrix.rows];
    bool *input = new bool[matrix.rows];
    int *result = new int[matrix.rows];
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
            if ((result[j] == 0) && (output[j] == true) && (source != j))
            {
                result[j] = dim;
                entry++;
            }
            input[j] = output[j];
            output[j] = false;
        }
        if ((entry > entry_last) && (entry < matrix.rows))
        {
            entry_last = entry;
            if (entry_last >= matrix.rows - 1) // entry = matrix.rows - 1意味着向量填满，无下一轮
                break;
        }
        else // 如果没有新的最短路径产生，则退出循环
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
        std::ofstream outfile(output_path);
        if (!outfile.is_open())
        {
            std::cerr << "Error opening file " << output_path << std::endl;
            return 0.0;
        }
        for (int j = 0; j < matrix.rows; j++)
        {
            if ((source != j) && (result[j] != 0))
                outfile << source << " " << j << " " << result[j] << endl;
        }
        outfile.close();
    }
    delete[] result;
    result = nullptr;

    return elapsed.count();
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

void DAWN::runApspGpu(DAWN::Matrix &matrix, std::string &output_path)
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return;
    }
    float elapsed_time = 0.0;
    int proEntry = 0;

    DAWN dawn;

    // Copy data to device
    int *d_A_entry, *d_A;
    cudaMalloc((void **)&d_A_entry, sizeof(int) * matrix.rows);
    cudaMalloc((void **)&d_A, sizeof(int) * matrix.nnz);

    cout << matrix.nnz << endl;
    int *h_A = new int[matrix.nnz];
    int tmp = 0;
    for (int i = 0; i < matrix.rows; i++)
    {
        if (matrix.A_entry[i] == 0)
        {
            matrix.A_entry[i] = matrix.A_entry[i - 1];
            continue;
        }
        for (int j = 0; j < matrix.A_entry[i]; j++)
        {
            h_A[tmp] = matrix.A[i][j];
            tmp++;
        }
        matrix.A_entry[i] = tmp;
    }

    std::cerr << "Copy Matrix" << std::endl;

    cudaMemcpy(d_A_entry, matrix.A_entry, sizeof(int) * matrix.rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, sizeof(int) * matrix.nnz, cudaMemcpyHostToDevice);

    // Create streams
    cudaStream_t streams[matrix.stream];
    for (int i = 0; i < matrix.stream; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < matrix.rows; i++)
    {

        int source = i;
        if (matrix.B_entry[i] == 0)
        {
            proEntry++;
            dawn.infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
            continue;
        }
        int cuda_stream = source % matrix.stream;
        float time_tmp = 0.0f;
        int *result = new int[matrix.rows];

        time_tmp = sssp_gpu(matrix, source, streams[cuda_stream], d_A_entry, d_A, result);
        elapsed_time += time_tmp;
        proEntry++;
        dawn.infoprint(proEntry, matrix.rows, matrix.interval, matrix.thread, elapsed_time);
        // Output
        if ((matrix.prinft == true) && (i == matrix.source))
        {
            for (int j = 0; j < matrix.rows; j++)
            {
                if (i != j && (result[j] > 1))
                    outfile << i << " " << j << " " << result[j] << endl;
            }
            delete[] result;
            result = nullptr;
        }
    }
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // Output elapsed time and free remaining resources
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << " Elapsed time: " << elapsed_time / (matrix.thread * 1000) << std::endl;

    outfile.close();

    // Synchronize streams
    for (int i = 0; i < matrix.stream; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Free memory on device
    cudaFree(d_A_entry);
    cudaFree(d_A);
}

void DAWN::runSsspGpu(DAWN::Matrix &matrix, std::string &output_path)
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

    // Copy data to device
    int *d_A_entry, *d_A;
    cudaMalloc((void **)&d_A_entry, sizeof(int) * matrix.rows);
    cudaMalloc((void **)&d_A, sizeof(int) * matrix.nnz);

    cout << matrix.nnz << endl;
    int *h_A = new int[matrix.nnz];
    int tmp = 0;
    for (int i = 0; i < matrix.rows; i++)
    {
        if (matrix.A_entry[i] == 0)
        {
            matrix.A_entry[i] = matrix.A_entry[i - 1];
            continue;
        }
        for (int j = 0; j < matrix.A_entry[i]; j++)
        {
            h_A[tmp] = matrix.A[i][j];
            tmp++;
        }
        matrix.A_entry[i] = tmp;
    }

    std::cerr << "Copy Matrix" << std::endl;

    cudaMemcpy(d_A_entry, matrix.A_entry, sizeof(int) * matrix.rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, sizeof(int) * matrix.nnz, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    int *result = new int[matrix.rows];
    float elapsed_time = sssp_gpu(matrix, source, stream, d_A_entry, d_A, result);

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    // Output elapsed time and free remaining resources
    std::cout << " Elapsed time: " << elapsed_time / (matrix.thread * 1000) << std::endl;

    // Output
    if (matrix.prinft == true)
        for (int j = 0; j < matrix.rows; j++)
        {
            if (source != j)
                outfile << source << " " << j << " " << result[j] << endl;
        }

    outfile.close();

    cudaStreamDestroy(stream);

    delete[] result;
    result = nullptr;
    // Free memory on device
    cudaFree(d_A_entry);
    cudaFree(d_A);
}

float DAWN::sssp_gpu(DAWN::Matrix &matrix, int source, cudaStream_t streams, int *d_A_entry, int *d_A, int *&result)
{

    int dim = 1;
    int entry = matrix.B_entry[source];
    int entry_last = entry;
    bool *output = new bool[matrix.rows];
    bool *input = new bool[matrix.rows];
    omp_set_dynamic(true);
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

    bool *d_input, *d_output;
    int *d_result;
    int *d_entry;

    cudaMalloc((void **)&d_input, sizeof(bool) * matrix.cols);
    cudaMalloc((void **)&d_output, sizeof(bool) * matrix.rows);
    cudaMalloc((void **)&d_result, sizeof(int) * matrix.rows);
    cudaMalloc(&d_entry, sizeof(int));

    cudaMemcpyAsync(d_input, input, sizeof(bool) * matrix.rows, cudaMemcpyHostToDevice, streams);
    cudaMemcpyAsync(d_output, output, sizeof(bool) * matrix.rows, cudaMemcpyHostToDevice, streams);
    cudaMemcpyAsync(d_entry, &entry, sizeof(int), cudaMemcpyHostToDevice, streams);
    cudaMemcpyAsync(d_result, result, sizeof(int) * matrix.rows, cudaMemcpyHostToDevice, streams);

    // Launch kernel
    int block_size = matrix.block_size;
    int num_blocks = (matrix.cols + block_size - 1) / block_size;
    int entry_max = matrix.rows - 1;

    auto start = std::chrono::high_resolution_clock::now();
    while (dim < matrix.dim)
    {
        dim++;
        vecMatOpe<<<num_blocks, block_size, 0, streams>>>(matrix.rows, d_A_entry, d_A, d_input, d_output, d_result, source, dim, d_entry);
        cudaMemcpyAsync(&entry, d_entry, sizeof(int), cudaMemcpyDeviceToHost, streams);
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

    cudaMemcpyAsync(result, d_result, sizeof(int) * matrix.rows, cudaMemcpyDeviceToHost, streams);
    matrix.entry += entry_last;

    delete[] output;
    output = nullptr;
    delete[] input;
    input = nullptr;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_result);
    cudaFree(d_entry);

    return elapsed.count();
}

__global__ void vecMatOpe(int rows, int *d_A_entry, int *d_A, bool *input, bool *output, int *result, int source, int dim, int *d_entry)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int entry = 0;
    if (j < rows && (d_A_entry[j] != d_A_entry[j - 1]))
    {

        int start = (j == 0 ? 0 : d_A_entry[j - 1]); // 当前行的起始位置
        int end = d_A_entry[j];                      // 当前行的结束位置
        for (int k = start; k < end; k++)
        {
            if (input[d_A[k]])
            {
                output[j] = true;
                if ((result[j] == 0) && (source != j))
                {
                    result[j] = dim;
                    ++entry;
                }
                break;
            }
        }
    }

    if (j < rows)
    {
        input[j] = output[j];
        output[j] = false;
    }
    atomicAdd(d_entry, entry);
}