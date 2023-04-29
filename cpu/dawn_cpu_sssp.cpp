#include "dawn.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    string algo = argv[1];
    DAWN dawn;
    DAWN::Matrix matrix;
    if (algo == "Default")
    {
        string input_path = argv[2];
        string output_path = argv[3];

        string prinft = argv[4];
        if (prinft == "true")
            matrix.prinft = true;
        else
            matrix.prinft = false;

        matrix.source = atoi(argv[5]);
        matrix.thread = 1;

        cout << "The source is " << matrix.source << endl;
        dawn.createGraph(input_path, matrix);
        dawn.runSsspCpu(matrix, output_path);
        return 0;
    }

    if (algo == "BIG")
    {
        string input_path = argv[2];
        string col_input_path = argv[3];
        string row_input_path = argv[4];
        string output_path = argv[5];

        string prinft = argv[6];
        if (prinft == "true")
            matrix.prinft = true;
        else
            matrix.prinft = false;

        matrix.source = atoi(argv[7]);
        matrix.thread = 1;
        cout << "The source is " << matrix.source << endl;
        dawn.readGraphBig(input_path, col_input_path, row_input_path, matrix);
        dawn.runSsspCpu(matrix, output_path);
        return 0;
    }

    return 0;
}