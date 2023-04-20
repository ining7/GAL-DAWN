#include "dawn.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    string col_input_path = argv[1];
    string row_input_path = argv[2];
    string output_path = argv[3];

    DAWN dawn;
    DAWN::Matrix matrix;
    matrix.thread = 20;
    matrix.interval = 10000;
    dawn.readGraphBig(col_input_path, row_input_path, matrix);
    dawn.runApspV3(matrix, output_path);
    return 0;
}
