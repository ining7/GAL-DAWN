# SC2023

1.Download testing data

Unzip the compressed package and put it in the directory you need

The input data can be found on the Science Data Bank

```c++
URL=https://www.scidb.cn/s/6BjM3a
GRAPH_DIR="to_your_graph_path"
```

2.RUN

```c++
cd $PROJECT_ROOT
mkdir build && cd build
cmake .. && make -j
```

If compilation succeeds without errors, you can run your code as before, for example

```c++
cd $PROJECT_ROOT/build
./dawn_cpu_v1 CG $GRAPH_DIR/mouse_gene.mtx ../outpu.txt 100 false 0

./dawn_cpu_v1 BCG $GRAPH_DIR/graph.mxt $GRAPH_DIR/graph_CRC.txt $GRAPH_DIR/graph_RCC.txt ../outpu.txt 10000 false 0

./convert $GRAPH_DIR/large_graph.mtx $GRAPH_DIR/graph_CRC.txt $GRAPH_DIR/graph_RCC.txt
```

When the version is built, it will SSSP applications, which can be used directly.

Please refer to decument/Decumention_v1 for commands.

If you need to use DAWN in your own solution, please check the source code of **dawn_cpu_sssp.cpp** and call it.

3.Using script.

```c++
cd ..
sudo vim ./process.sh
MAIN = ${main}
GRAPH_DIR = ${test_graph}
OUTPUT= ${outputfile}
LOG_DIR= ${GRAPH_DIR}/log
ESC && wq
sudo chmod +x ../process.sh
sudo bash ../process.sh
```

Please note that the normal operation of the batch script needs to ensure that the test machine meets the minimum requirements. Insufficient memory or GPU memory needs to be manually adjusted according to amount of resources.

```c++
CPU: Multi-threaded processor supporting OpenMP API
RAM: 8GB or more
Compiler: GCC 9.4.0 and above, clang 10.0.0 and above.
OS:  Ubuntu 20.04 and above
```
