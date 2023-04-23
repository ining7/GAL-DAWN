#include "dawn.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    string input_path = argv[1];
    string output_path = argv[2];

    DAWN dawn;
    DAWN::Matrix matrix;
    matrix.thread = 20;
    matrix.interval = 1;
    dawn.createGraph(input_path, matrix);
    dawn.runApspV4(matrix, output_path);
    return 0;
}

// bool Del_Min(SqList &L, ElemType &value)
// {
//     if (L.length == 0)
// }
