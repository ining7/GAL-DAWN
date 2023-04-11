#!/bin/bash

# Use "sudo bash process_full_sssp.sh" to run the script

# Modify the absolute path of MAIN and GRAPH_DIR, or the relative path based on the directory where it is located.
MAIN="/home/lxr/ining/SC2023/ining_final/build/vtov" #需要修改为demo地址
GRAPH_DIR="/home/lxr/sc2023/gunrock/test_graph"
output="/home/lxr/ining/SC2023/ining_final/v2v_check.txt"

# Set directory path for the graph log files
LOG_DIR="${GRAPH_DIR}/vtov_log"


# Check if the GRAPH_DIR path exists and contains any mtx files
if [[ ! -d "${GRAPH_DIR}" ]]; then
    echo "Error: ${GRAPH_DIR} does not exist or is not a directory!"
    exit 1
fi

# Create LOG_DIR if it doesn't exist already
[[ ! -d "${LOG_DIR}" ]] && mkdir "${LOG_DIR}"


# Loop over all mtx files in GRAPH_DIR directory
for file in ${GRAPH_DIR}/*.mtx; do
    if [[ ! -f "${file}" ]]; then
        continue
    fi
    
    # Extract filename from filepath, without .mtx extension
    filename=$(basename -- "${file}")
    filename="${filename%.*}"
    echo "Proccessing ${file}!"
    # Run full_sssp on the mtx file and redirect output to logfile
    "${MAIN}" "${file}" "${output}" | tee "${LOG_DIR}/${filename}_log.txt"
    echo "${MAIN}" "${file}" "${output}" | tee "${LOG_DIR}/${filename}_log.txt"
    echo "${file} proccessed, please check the log file for details. log files: ${LOG_DIR}/${filename}_log.txt"
done

echo "All done!"

