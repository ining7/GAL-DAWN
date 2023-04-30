#include "dawn.cuh"

int main(int argc, char *argv[])
{
    DAWN dawn;
    DAWN::Matrix matrix;
    string algo = argv[1];
    string input_path = argv[2];
    if (algo == "Default")
    {
        string output_path = argv[3];
        matrix.stream = atoi(argv[4]);
        matrix.block_size = atoi(argv[5]);
        matrix.interval = atoi(argv[6]);
        string prinft = argv[7];
        matrix.source = atoi(argv[8]);

        if (prinft == "true")
        {
            matrix.prinft = true;
            cout << "Prinft source " << matrix.source << endl;
        }
        else
            matrix.prinft = false;

        matrix.thread = 1;
        std::ifstream file(input_path);
        if (!file.is_open())
        {
            std::cerr << "Error opening file " << input_path << std::endl;
            return;
        }
        dawn.createGraph(input_path, matrix);
        dawn.runApspGpu(matrix, output_path);
    }

    if (algo == "Big")
    {
        string col_input_path = argv[3];
        string row_input_path = argv[4];
        string output_path = argv[5];
        matrix.stream = atoi(argv[6]);
        matrix.block_size = atoi(argv[7]);
        matrix.interval = atoi(argv[8]); // 请保证打印间隔小于节点总数,建议10000
        string prinft = argv[9];
        if (prinft == "true")
            matrix.prinft = true;
        else
            matrix.prinft = false;
        matrix.source = atoi(argv[10]);

        matrix.thread = 1;

        dawn.readGraphBig(input_path, col_input_path, row_input_path, matrix);
        dawn.runApspGpu(matrix, output_path);
        return 0;
    }

    return 0;
}