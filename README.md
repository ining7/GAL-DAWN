# DAWN: An Noval SSSP/APSP Algorithm, CUDA/C++

DAWN is a novel shortest paths algorithm, which is suitable for weighted and unweighted graphs. In contrast to the prevalent optimization of state-of-the-art BFS implementation, which commonly rely on priority queues, our approach leverages matrix operations to endow DAWN with enhanced parallelism. DAWN is capable of solving the shortest path problems on graphs with negative weights, and can automatically exclude the influence of negative weight cycles.

DAWN requires $O(m)$ space and $O(S_{wcc} \cdot E_{wcc})$ times on the unweighted graphs, which can also process SSSP tasks and requires $O(E_{wcc}(i))$ time. $S_{wcc}$ and $E_{wcc}$ denote the number of nodes and edges included in the largest WCC (Weakly Connected Component) in the graphs.

| [**Examples**](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/tree/dev/algorithm) | [**Documentation**](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/tree/dev/document) | [**Test**](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/tree/dev/test) |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
- Examples: Demonstrate the usage of algorithms in DAWN.
- Document: Provides the detailed description of DAWN, include the **Quick_Start** and **Code_Guide**. **Quick_Start** provides the guide for quickly working with DAWN. **Code_Guide** provides the detailed description for how to implement own algorithm in DAWN.
- Test: Provides the detailed information of Testing.


## Development Status

The betweenness centrality implementation based on DAWN is currently under development, and CPU version has appeared in the dev branch. We have utilized an accumulation technique and graph traversal approach of DAWN that differs from the Brandes algorithm, which is a novel algorithm with lower time and space complexity, tentatively named BCDAWN. Moving forward, our focus will be on parallelizing the current CPU version and developing a GPU version. We plan to release a paper describing the technical details of the weighted version of DAWN and the BCDAWN algorithm at an appropriate time, and these implementations will be made available for early access. We still encourage colleagues to complete the implementation of the Brandes algorithm.

We welcome any interest and ideas related to DAWN and its applications. If you are interested in DAWN algorithms and their applications, please feel free to share your thoughts via [email](<1289539524@qq.com>), and we will do our best to assist you in your research based on DAWN.

The repository contributors and paper authors hold all rights to the code currently in the repository and the corresponding publications. If you are interested in work related to the DAWN algorithm and are developing new algorithms, please be mindful of not replicating the technical implementations currently in our repository. If you are working on engineering optimizations for algorithms like DAWN, please promptly create a fork and raise issues in this repository to avoid instances of plagiarism and preemptive paper publication.

| Algorithm           | Release |
| ------------------- | ------- |
| APSP                | V2.1    |
| MSSP                | V2.1    |
| SSSP                | V2.1    |
| BFS                 | V2.1    |
| BC                  | Doing   |
| CC                  | V2.3    |
| Cluster Analysis    | Future  |
| Community Detection | Future  |

Further applications of these algorithms, such as community detection, clustering, and path planning, are also on our agenda.

The DAWN component based on Gunrock may be released to the main/develop branch in the near future, so please stay tuned to the [Gunrock](https://github.com/gunrock/gunrock). We will release new features of DAWN and the application algorithms based on DAWN on this repository. If the algorithms are also needed by Gunrock, we will contribute them to the Gunrock repository later.

## How to Cite DAWN
Thank you for citing our work. 

```bibtex
@InProceedings{Feng:2024:DAWN,
  author =	 {Yelai Feng and Huaixi Wang and Yining Zhu and Xiandong Liu and Hongyi Lu and Qing Liu},
  title =	 {DAWN: Matrix Operation-Optimized Algorithm for Shortest Paths Problem on Unweighted Graphs},
  booktitle =	 {Proceedings of the 38th ACM International Conference on Supercomputing},
  year =	 {2024},
  doi =		 {10.1145/3650200.3656600}
}
```

## Copyright & License

All source code are released under [Apache 2.0](https://github.com/lxrzlyr/DAWN-An-Noval-SSSP-APSP-Algorithm/blob/4266d98053678ce76e34be64477ac2364f0f4291/LICENSE).
