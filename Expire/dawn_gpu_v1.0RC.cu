#include "dawn.cuh"

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string output_path = argv[2];
    int stream = atoi(argv[3]);
    int block_size = atoi(argv[4]);
    int interval = atoi(argv[5]);
    int prinft = atoi(argv[6]);

    std::ifstream file(input_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << input_path << std::endl;
        return;
    }

    DAWN dawn;
    DAWN::Graph matrix;
    matrix.thread = 1;          // 运行SSSP的线程,GPU版本默認爲1
    matrix.interval = interval; // 100
    matrix.stream = stream;     // 32
    matrix.block_size = block_size;
    matrix.prinft = prinft;

    dawn.createGraph(input_path, matrix);
    dawn.runApspGpu(matrix, output_path);

    return 0;
}