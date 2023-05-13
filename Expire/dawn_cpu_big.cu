#include "dawn.hpp"
using namespace std;

int main(int argc, char *argv[])
{

    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " ./input_path.mtx ./col_input_path.txt ./row_input_path.txt ./output_path.txt" << std::endl;
        return 0;
    }

    string input_path = argv[1];
    string col_input_path = argv[2];
    string row_input_path = argv[3];
    string output_path = argv[4];

    DAWN dawn;
    DAWN::Graph matrix;
    matrix.thread = 1;
    matrix.interval = 10000;
    dawn.readGraphBig(input_path, col_input_path, row_input_path, matrix);
    dawn.runApspV3(matrix, output_path);
    return 0;
}
