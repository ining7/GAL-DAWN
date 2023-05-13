#include "dawn.cuh"
using namespace std;

int main(int argc, char *argv[])
{
    string algo = argv[1];
    DAWN dawn;
    DAWN::Matrix matrix;
    if ((algo == "FG") || (algo == "CG"))
    {
        string input_path = argv[2];
        string output_path = argv[3];
        matrix.interval = atoi(argv[4]); // 请保证打印间隔小于节点总数，建议10-1000
        string prinft = argv[5];
        matrix.source = atoi(argv[6]);
        if (prinft == "true")
        {
            matrix.prinft = true;
            cout << "Prinft source " << matrix.source << endl;
        }
        else
            matrix.prinft = false;

        if (algo == "CG")
        {
            matrix.thread = 20;
            dawn.createGraph(input_path, matrix);
            dawn.runApspV4(matrix, output_path);
            return 0;
        }
        else
        {
            matrix.thread = 1;
            dawn.createGraph(input_path, matrix);
            dawn.runApspV3(matrix, output_path);
            return 0;
        }
    }
    if ((algo == "BFG") || (algo == "BCG"))
    {
        string input_path = argv[2];
        string col_input_path = argv[3];
        string row_input_path = argv[4];
        string output_path = argv[5];
        matrix.interval = atoi(argv[6]); // 请保证打印间隔小于节点总数,建议10000
        string prinft = argv[7];
        if (prinft == "true")
            matrix.prinft = true;
        else
            matrix.prinft = false;
        matrix.source = atoi(argv[8]);
        if (algo == "BCG")
        {
            matrix.thread = 20;
            cout << "matrix.source: " << matrix.source << endl;
            dawn.readGraphBig(input_path, col_input_path, row_input_path, matrix);
            dawn.runApspV4(matrix, output_path);
            return 0;
        }
        else
        {
            matrix.thread = 1;
            cout << "matrix.source: " << matrix.source << endl;
            dawn.readGraphBig(input_path, col_input_path, row_input_path, matrix);
            dawn.runApspV3(matrix, output_path);
            return 0;
        }
    }

    return 0;
}
