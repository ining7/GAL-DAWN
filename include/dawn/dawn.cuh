#include <dawn/dawn.hxx>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

namespace DAWN {
class GPU {
public:
  void runAPSPGpu(Graph& graph, std::string& output_path);

  void runAPBFSGpu(Graph& graph, std::string& output_path);

  void runMBFSGpu(Graph& graph, std::string& output_path);

  void runMSSPGpu(Graph& graph, std::string& output_path);

  void runSSSPGpu(Graph& graph, std::string& output_path);

  void runBFSGpu(Graph& graph, std::string& output_path);

  float BFSGpu(Graph&                     graph,
               int                        source,
               cudaStream_t               streams,
               thrust::device_vector<int> d_row_ptr,
               thrust::device_vector<int> d_col,
               std::string&               output_path);

  float SSSPGpu(Graph&                       graph,
                int                          source,
                cudaStream_t                 streams,
                thrust::device_vector<int>   d_row_ptr,
                thrust::device_vector<int>   d_col,
                thrust::device_vector<float> d_val,
                std::string&                 output_path);
};
}  // namespace DAWN
