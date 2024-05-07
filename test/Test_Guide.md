# Introduction
We have presented a performance comparison of algorithms for DAWN, GAPBS, and Gunrock in [Performance](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/tree/dev/test/Performance.md). The benchmark tests were run on the Gunrock benchmark dataset and the Suite Sparse Collection dataset. The table provides specific information about the graphs and their corresponding runtime. The baseline implementations from Gunrock and GAPBS are provided in the **test** directory.

# Test Environment

The test environment is as follows:

- OS: Ubuntu 20.04.5 LTS
- CPU: Intel Core i5-13600KF
- GPU: NVIDIA GeForce RTX 2080 Ti
- Memory: 32GB
- CUDA: 12.1

# Code

We also provide the test code for Gunrock in the **test/gunrock** and GAPBS in the **test/gapbs**. Due to differences in code build environments and other aspects among the repositories, it is not possible to pull and build them uniformly. Alternatively, you can pull our modified fork branch and build directly([Gunrock](https://github.com/lxrzlyr/gunrock),[GAPBS](https://github.com/lxrzlyr/gapbs)).

If you need to verify the results of Gunrock and GAPBS, please visit the repositories for [Gunrock](https://github.com/gunrock/gunrock) and [GAPBS](https://github.com/sbeamer/gapbs) respectively, follow the repository build instructions, and replace the source files in the repository with the ones we provide.

# Check the Results
We provide the file **check_unweighted.py** and **check_weighted.py**, based on networkx, which can be used to check the results printed by DAWN.