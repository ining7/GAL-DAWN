#include "dawn.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string output_path = argv[2];

    DAWN dawn;
    DAWN::Matrix matrix;
    matrix.thread = 20;
    matrix.interval = 100;
    dawn.createGraph(input_path, matrix);
    dawn.runApspV3(matrix, output_path);
    return 0;
}
