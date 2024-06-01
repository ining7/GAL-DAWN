/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-05-07
 *
 * @copyright Copyright (c) 2024
 */
#!/bin/bash

# Modify the absolute path of MAIN and GRAPH_DIR, or the relative path based on the directory where it is located.
MAIN="${PROJECT_ROOT}/build"
GRAPH_DIR=""
SourceList=""
OUTPUT="output.txt"
Interval="100"
Prinft="false"
Stream="4"
Block_size="1024"
Weighted="unweighted"

# Check if the GRAPH_DIR path exists and contains any mtx files
if [[ ! -d "${GRAPH_DIR}" ]]; then
    echo "Error: ${GRAPH_DIR} does not exist or is not a directory!"
    exit 1
fi

# Set directory path for the graph log files
LOG_DIR=""

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
    echo "Proccessing ${file}! Please check the log file for details."
    # Use the command you need.
    # test cpu
    "${MAIN}/mssp_cpu" "${file}" "${OUTPUT}" "${Prinft}" "${SourceList}" "${Weighted}"| tee "${LOG_DIR}/${filename}_log.txt" 
    # test gpu
    "${MAIN}/mssp_gpu" "${file}" "${OUTPUT}" "${Stream}" "${Block_size}" "${Prinft}" "${SourceList}" "${Weighted}" | tee "${LOG_DIR}/${filename}_log.txt" 
done

echo "All done!"

