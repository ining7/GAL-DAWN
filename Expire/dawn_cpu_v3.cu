#include "dawn.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string output_path = argv[2];

    DAWN dawn;
    DAWN::Matrix matrix;
    matrix.thread = 1;   // 运行SSSP的线程
    matrix.interval = 1; // 请保证打印间隔小于节点总数
    dawn.createGraph(input_path, matrix);
    dawn.runApspV3(matrix, output_path);
    return 0;
}
