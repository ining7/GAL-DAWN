#include "dawn.cuh"
using namespace std;

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string output_path = argv[2];
    int block_size = atoi(argv[3]);
    int prinft = atoi(argv[4]);

    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }

    DAWN dawn;
    DAWN::Matrix matrix;
    matrix.block_size = block_size;
    matrix.prinft = prinft;
    matrix.thread = 1;

    dawn.createGraph(input_path, matrix);
    dawn.runSsspGpu(matrix, output_path);

    return 0;
}