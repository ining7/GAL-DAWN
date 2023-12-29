import sys
import numpy as np
import scipy.sparse
import scipy.io
from scipy.sparse import csc_matrix

def test_sssp(num_arguments, argument_array):
    if num_arguments != 3:
        print("usage: ./bin/<program-name> filename.mtx sourceList.txt")
        sys.exit(1)

    # Define types
    vertex_t = int
    edge_t = int
    weight_t = float

    # IO
    filename = argument_array[1]
    sourceList = argument_array[2]

    if filename.endswith('.mtx'):
        # Load data from a Matrix Market file
        csr = scipy.io.mmread(filename).tocsr()
    elif filename.endswith('.bin'):
        # Load data from a binary CSR file (You'll need to implement this part)
        print("Binary CSR file format not supported.")
        sys.exit(1)
    else:
        print("Unknown file format: " + filename)
        sys.exit(1)

    with open(sourceList, 'r') as file:
        sourcelist = [int(line) for line in file if not line.startswith('%')]

    # Initialize nonzero_values with 1.0
    nonzero_values = np.ones(csr.nnz, dtype=weight_t)
    csr.data = nonzero_values

    # Build graph
    G = csc_matrix(csr)

# Main program
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python <script_name>.py filename.mtx sourceList.txt")
        sys.exit(1)
    
    test_sssp(len(sys.argv), sys.argv)
