#include "dawn.cuh"
using namespace std;

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string col_output_path = argv[2];
    string row_output_path = argv[3];

    DAWN dawn;
    DAWN::Matrix matrix;
    matrix.thread = 20;
    matrix.interval = 1;
    dawn.createGraphconvert(input_path, matrix, col_output_path, row_output_path);
    return 0;
}
