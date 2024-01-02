#!/bin/bash

# Use "sudo bash process_full_sssp.sh" to run the script

# Modify the absolute path of MAIN and GRAPH_DIR, or the relative path based on the directory where it is located.
MAIN="/home/lxr/code/SC2023/build/test" #需要修改为main地址
GRAPH_DIR="/home/lxr/code/test_graph_gunrock"
OUTPUT="/home/lxr/code/SC2023/out.txt"
Algorithm="Mssp"
Interval="100"
Prinft="false"
Source="0"
SourceList="/home/lxr/code/SC2023/sourceList.txt"
Stream="4"
Block_size="32"
Weighted="unweighted"

# Check if the GRAPH_DIR path exists and contains any mtx files
if [[ ! -d "${GRAPH_DIR}" ]]; then
    echo "Error: ${GRAPH_DIR} does not exist or is not a directory!"
    exit 1
fi

# Set directory path for the graph log files
LOG_DIR="/home/lxr/code/SC2023/log/test_log"

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
    # Run full_sssp on the mtx file and redirect output to logfile
    # "${MAIN}" "${Algorithm}" "${file}" "${OUTPUT}" "${Interval}" "${Prinft}" "${SourceList}" "${Weighted}"| tee "${LOG_DIR}/${filename}_log.txt" #cpu
    # "${MAIN}" "${Algorithm}" "${file}" "${OUTPUT}" "${Stream}" "${Block_size}" "${Interval}" "${Prinft}" "${Source}" | tee "${LOG_DIR}/${filename}_log.txt" #gpu
    # "${MAIN}" "${Algorithm}" "${file}" "${OUTPUT}" "${Block_size}" "${Prinft}" "${SourceList}" "${Weighted}"| tee "${LOG_DIR}/${filename}_log.txt" #mssp
    "${MAIN}" "${Algorithm}" "${file}" "${OUTPUT}" "${SourceList}" | tee "${LOG_DIR}/${filename}_log.txt" #test
done

echo "All done!"

