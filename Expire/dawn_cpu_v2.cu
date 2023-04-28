#include "access.h"
#include "omp.h"

using namespace std;

struct Matrix
{
    int rows;
    int cols;
    uint64_t nnz;
    bool **input;
    int **dense;
    int *dense_entry;
    bool **result;
    int *entry;
    int dim;
};

float runDawn(Matrix &matrix, string &input_path, string &output_path);
void readgraph(string &input_path, Matrix &matrix);

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

    matrix.result = new bool *[matrix.rows];
    matrix.input = new bool *[matrix.rows];
    matrix.dense = new int *[matrix.rows];
    matrix.dense_entry = new int[matrix.rows];
    matrix.entry = new int[matrix.rows];
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        matrix.result[i] = new bool[matrix.rows];
        matrix.input[i] = new bool[matrix.rows];
        matrix.dense_entry[i] = -1;
        matrix.entry[i] = 0;
        for (int j = 0; j < matrix.rows; j++)
        {
            matrix.result[i][j] = false;
            matrix.input[i][j] = false;
        }
    }
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        ss >> rows >> cols;
        rows--;
        cols--;
        if (rows != cols)
        {
            matrix.result[rows][cols] = true;
            matrix.input[rows][cols] = true;
        }
    }
    file.close();

#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        vector<int> tmp;
        tmp.clear();
        int entry_tmp = 0;
        for (int j = 0; j < matrix.cols; j++)
        {
            if (matrix.result[j][i] == true && i != j)
            {
                tmp.push_back(j);
                entry_tmp++;
            }
            if (matrix.input[i][j] == true && i != j)
            {
                matrix.entry[i]++;
            }
        }
        if (entry_tmp > 0)
        {
            matrix.dense_entry[i] = entry_tmp;
        }
        matrix.dense[i] = new int[entry_tmp];
        for (int j = 0; j < entry_tmp; j++)
        {
            matrix.dense[i][j] = tmp[j];
        }
    }
    cout << "Initialize input matrices" << endl;
}

float runDawn(Matrix &matrix, string &input_path, string &output_path)
{
    readgraph(input_path, matrix);

    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening file " << output_path << std::endl;
        return 0.0;
    }

    int dim = 1;

    float elapsed_time = 0;
    uint64_t entry_total = 0;
    uint64_t entry_total_last = 0;
    bool *tmp_output = new bool[matrix.rows];
    int *entry = new int[matrix.rows];
#pragma omp parallel for
    for (int i = 0; i < matrix.rows; i++)
    {
        tmp_output[i] = false;
        if (matrix.dense_entry[i] == -1)
            continue;
        for (int j = 0; j < matrix.dense_entry[i]; j++)
        {
            // outfile << matrix.dense[i][j] << " " << i << " " << dim << endl;
            entry[i]++;
        }
    }
    for (int i = 0; i < matrix.rows; i++)
    {
        entry_total += entry[i];
        entry[i] = 0;
    }

    cout << "Path length 1:" << entry_total << endl;
    entry_total_last = entry_total;

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    while (dim < matrix.dim)
    {

        dim++;
        // MAINFUNCTION
        for (int i = 0; i < matrix.rows; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            if ((matrix.dense_entry[i] == -1) || (matrix.entry[i] == -1))
            {
                // cout << " i : " << i << endl;
                continue;
            }

#pragma omp parallel for
            for (int j = 0; j < matrix.rows; j++)
            {
                for (int k = 0; k < matrix.dense_entry[j]; k++)
                {
                    if (matrix.input[i][matrix.dense[j][k]] == true)
                    {
                        tmp_output[j] = true;
                        break;
                    }
                }
            }

#pragma omp parallel for
            for (int j = 0; j < matrix.rows; j++)
            {
                if (matrix.result[i][j] == false && tmp_output[j] == true && i != j)
                {
                    matrix.result[i][j] = true;
                    entry[j]++;
                    // outfile << i << " " << j << " " << dim << endl;
                }
                matrix.input[i][j] = tmp_output[j];
                tmp_output[j] = false;
            }
            uint64_t entry_total_tmp = 0;

            for (int j = 0; j < matrix.rows; j++)
            {
                entry_total_tmp += entry[j];
                entry[j] = 0;
            }

            if (entry_total_tmp > 0)
            {
                matrix.entry[i] += entry_total_tmp;
                entry_total += entry_total_tmp;
                if (matrix.entry[i] > matrix.rows - 1)
                    break;
            }
            else
            {
                matrix.entry[i] = -1;
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            elapsed_time += elapsed.count();
        }
        std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;
        cout << "dim : " << dim << endl;
        cout << "entry : " << entry_total << endl;

        // if (entry_total > matrix.rows * (matrix.rows - 1) - 1)
        //     break;
        if (entry_total > entry_total_last)
            entry_total_last = entry_total;
        else
        {
            cout << " entry_total: " << entry_total << endl;
            cout << " entry_total_last: " << entry_total_last << endl;
            break;
        }
    }

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    outfile.close();
    delete[] tmp_output;
    return elapsed_time;
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

    float elapsed_time = 1.0f * runDawn(matrix, input_path, output_path);

    // Output elapsed time and free remaining resources
    std::cout << " Elapsed time: " << elapsed_time / 1000 << std::endl;

    return 0;
}
