#!/bin/bash

# This script monitors CPU, memory, and GPU usage and saves it into cpu_mem_usage.csv and gpu_usage.csv.

export LC_NUMERIC="C"

cpu_mem_output_file="cpu_mem_usage.csv"
gpu_output_file="gpu_usage.csv"

# Check if nvidia-smi is available for GPU monitoring
if ! command -v nvidia-smi &> /dev/null
then
    echo "nvidia-smi could not be found, GPU monitoring will not be available."
    gpu_monitoring=0
else
    gpu_monitoring=1
fi

# Initialize the CPU and memory usage CSV file with headers
echo "Timestamp,CPU Usage (%),Memory Usage (%)" > "$cpu_mem_output_file"

# Initialize the GPU usage CSV file with headers
echo "Timestamp,GPU Number,GPU Usage (%)" > "$gpu_output_file"

# Function to get CPU and memory usage
get_cpu_mem_usage() {
    # Extract CPU usage as the complement of idle CPU from the 4th column of top's output
    cpu=$(mpstat 1 1 | awk '/Average:/ {print 100 - $12}' | tr -d '\n')
    # Extract memory usage as the percentage of used memory
    mem=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}' | tr -d '\n')
    # Ensure the output is in one line without extra spaces
    echo "$cpu,$mem"
}

# Function to get GPU usage
get_gpu_usage() {
    if [ "$gpu_monitoring" -eq 1 ]; then
        nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits
    else
        echo "N/A,N/A"
    fi
}

# Loop to collect data every second until the script is stopped
while true
do
    # Get the current timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Get CPU and memory usage
    cpu_mem_usage=$(get_cpu_mem_usage)
    # Append the data to the CPU and memory usage CSV file
    echo "$timestamp,$cpu_mem_usage" >> "$cpu_mem_output_file"

    # Get GPU usage
    gpu_usage=$(get_gpu_usage)
    # Append the data to the GPU usage CSV file with timestamp
    while IFS= read -r line
    do
        echo "$timestamp,$line" >> "$gpu_output_file"
    done <<< "$gpu_usage"

    # Wait for a second
    sleep 1
done
