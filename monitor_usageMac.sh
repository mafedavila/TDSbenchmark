#!/bin/bash

# This script monitors CPU and memory usage on macOS and saves it into cpu_mem_usage.csv.

export LC_NUMERIC="C"

cpu_mem_output_file="cpu_mem_usage.csv"

# Initialize the CPU and memory usage CSV file with headers
echo "Timestamp,CPU Usage (%),Memory Usage (%)" > "$cpu_mem_output_file"

# Function to get CPU usage
get_cpu_usage() {
    # Extract CPU usage from top's output
    cpu=$(top -l 1 -n 0 | grep -E "^CPU" | awk '{print $3}' | sed 's/%//')
    echo "$cpu"
}

# Function to get memory usage
get_mem_usage() {
    # Extract memory usage from vm_stat's output
    pages_free=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    pages_used=$(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
    pages_spec=$(vm_stat | grep "Pages speculative" | awk '{print $3}' | sed 's/\.//')
    pages_wired=$(vm_stat | grep "Pages wired down" | awk '{print $4}' | sed 's/\.//')
    pages_inactive=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
    pages_total=$(sysctl -n hw.memsize | awk '{print $1/4096}')
    mem_used=$((pages_used + pages_wired + pages_inactive + pages_spec))
    mem_usage=$(echo "scale=2; ($mem_used/$pages_total) * 100" | bc)
    echo "$mem_usage"
}

# Loop to collect data every second until the script is stopped
while true
do
    # Get the current timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Get CPU and memory usage
    cpu_usage=$(get_cpu_usage)
    mem_usage=$(get_mem_usage)

    # Append the data to the CPU and memory usage CSV file
    echo "$timestamp,$cpu_usage,$mem_usage" >> "$cpu_mem_output_file"

    # Wait for a second
    sleep 1
done