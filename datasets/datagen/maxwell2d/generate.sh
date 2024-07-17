#!/bin/bash

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--num_points) N="$2"; shift ;;
        -p|--partition) partition="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Directory to save the data
DIR_PATH="../../data/maxwell2d/$partition"
mkdir -p "$DIR_PATH"

# Function to check if a command exists (will check if GNU Parallel is installed)
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if GNU Parallel is installed, if not, fall back to sequential execution
if command_exists parallel; then
    echo "GNU Parallel is installed. Proceeding with parallel execution."
    seq 0 $((N-1)) | parallel -j+0 python main.py --dir_path "$DIR_PATH" --idx {}
else
    echo "GNU Parallel is not installed. Falling back to sequential execution."
    echo "To install GNU Parallel on Linux, run:"
    echo "sudo apt-get install parallel"
    echo "This might significantly speed up the data generation process."
    echo ""
    echo "Proceeding with sequential execution..."
    for i in $(seq 0 $((N-1))); do
        python main.py --dir_path "$DIR_PATH" --idx $i
    done
fi

echo "All processes completed."