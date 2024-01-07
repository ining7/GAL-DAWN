0. Before getting started

# Depending on your GPU, you may also want to edit the CUDA_ARCHITECTURES variable in $PROJECT_ROOT/CMakeLists.txt

# We use RTX 3080ti for computing, so CUDA_ARCHITECTURES is set to 80.

# Please read the gunrock introduction in https://github.com/gunrock/gunrock/wiki/How-to-write-a-new-graph-algorithm

export PROJECT_ROOT=/home/lxr/sc2023/gunrock

1. Modify $PROJECT_ROOT/examples/CMakeLists.txt

# To tell the build system about apsp, add the following line to $PROJECT_ROOT/examples/CMakeLists.txt:

add_subdirectory(apsp)

2. Create examples/apsp/CMakeLists.txt

cd $PROJECT_ROOT

# create directory

mkdir $PROJECT_ROOT/examples/apsp

# copy boilerplate CMakeLists.txt to `examples/apsp`

cp $PROJECT_ROOT/examples/algorithms/bfs/CMakeLists.txt $PROJECT_ROOT/examples/apsp/CMakeLists.txt

# change APPLICATION_NAME from `bfs` to `apsp`

sed -i "s/set(APPLICATION_NAME bfs)/set(APPLICATION_NAME apsp)/" $PROJECT_ROOT/examples/apsp/CMakeLists.txt

3. Create $PROJECT_ROOT/examples/apsp/apsp.cu

4. Create $PROJECT_ROOT/include/gunrock/algorithms/apsp.hxx

# We use sssp.hxx here.

5. RUN

cd $PROJECT_ROOT
mkdir build
cd build
cmake ..
make apsp -j

# If compilation succeeds without errors, you can run your code as before:

cd $PROJECT_ROOT/build
./bin/apsp ../datasets/chesapeake/chesapeake.mtx
